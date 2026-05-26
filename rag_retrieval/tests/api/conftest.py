from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval import RetrievalHit


SAMPLE_METADATA: List[Dict[str, Any]] = [
    {
        "id": "chunk-001",
        "document_id": "kr155_2",
        "document_title": "КР155 Ишемическая болезнь сердца",
        "section_id": "2",
        "section_title": "Диагностика",
        "label": "diagnosis",
        "specialty": "cardiology",
        "page_start": 12,
        "page_end": 13,
        "source": "docs/kr155_2.pdf",
        "text": "Диагноз ИБС устанавливается на основании клинической картины.",
        "embedding_text": "passage: Диагноз ИБС устанавливается на основании клинической картины.",
        "chunk_index": 5,
        "content_hash": "abc123",
    },
    {
        "id": "chunk-002",
        "document_id": "kr155_2",
        "document_title": "КР155 Ишемическая болезнь сердца",
        "section_id": "3",
        "section_title": "Лечение",
        "label": "treatment",
        "specialty": "cardiology",
        "page_start": 20,
        "page_end": 22,
        "source": "docs/kr155_2.pdf",
        "text": "Применяется антиангинальная терапия.",
        "embedding_text": "passage: Применяется антиангинальная терапия.",
        "chunk_index": 6,
        "content_hash": "def456",
    },
    {
        "id": "chunk-003",
        "document_id": "kr200_1",
        "document_title": "КР200 Гипертоническая болезнь",
        "section_id": "1",
        "section_title": "Общие положения",
        "label": "general",
        "specialty": "therapy",
        "page_start": 1,
        "page_end": 3,
        "source": "docs/kr200_1.pdf",
        "text": "Артериальная гипертония является хроническим заболеванием.",
        "embedding_text": "passage: Артериальная гипертония является хроническим заболеванием.",
        "chunk_index": 0,
        "content_hash": "ghi789",
    },
]


def make_hit(
    rank: int,
    chunk_id: str,
    document_id: str = "kr155_2",
    document_title: str = "КР155",
    section_id: str = "2",
    section_title: str = "Диагностика",
    label: str = "diagnosis",
    source: str = "docs/kr155_2.pdf",
    page_start: int = 12,
    page_end: int = 13,
    text: str = "Текст фрагмента.",
    score: float = 0.9,
    specialty: str = "cardiology",
    chunk_index: int = 5,
    context_source: str = "reranker",
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_title,
        section_id=section_id,
        section_title=section_title,
        label=label,
        source=source,
        page_start=page_start,
        page_end=page_end,
        text=text,
        extra={
            "specialty": specialty,
            "chunk_index": chunk_index,
            "context_source": context_source,
            "content_hash": "abc",
        },
    )


@pytest.fixture()
def sample_hits() -> List[RetrievalHit]:
    return [
        make_hit(rank=1, chunk_id="chunk-001", score=0.95, context_source="anchor"),
        make_hit(
            rank=2, chunk_id="chunk-002", score=0.82,
            document_id="kr155_2", label="treatment",
            section_title="Лечение", page_start=20, page_end=22,
            context_source="reranker",
        ),
        make_hit(
            rank=3, chunk_id="chunk-003", score=0.71,
            document_id="kr200_1", document_title="КР200",
            label="general", specialty="therapy",
            page_start=1, page_end=3,
            context_source="reranker_fallback",
        ),
    ]


@pytest.fixture()
def sample_metadata() -> List[Dict[str, Any]]:
    return SAMPLE_METADATA


@pytest.fixture()
def sample_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    emb = rng.random((3, 1024), dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture()
def mock_retrieval_service():
    from src.api.retrieval_service import RetrievalService

    svc = MagicMock(spec=RetrievalService)
    svc.is_loaded = True

    svc.get_health.return_value = {
        "status": "ok",
        "service": "rag-retrieval-api",
        "loaded": True,
        "embedding_model_key": "e5_large",
        "reranker_key": "bge_reranker_v2_m3",
        "num_chunks": 3702,
        "device": "cpu",
        "uptime_sec": 10.0,
    }
    svc.get_ready.return_value = {
        "ready": True,
        "checks": {
            "embeddings_loaded": True,
            "metadata_loaded": True,
            "embedding_model_loaded": True,
            "reranker_loaded": True,
        },
    }
    svc.stats.return_value = {
        "num_chunks": 3,
        "num_documents": 2,
        "labels": {"diagnosis": 1, "treatment": 1, "general": 1},
        "specialties": {"cardiology": 2, "therapy": 1},
        "embedding_dim": 1024,
    }
    svc.documents.return_value = [
        {
            "document_id": "kr155_2",
            "document_title": "КР155 Ишемическая болезнь сердца",
            "num_chunks": 2,
            "labels": ["diagnosis", "treatment"],
            "page_min": 12,
            "page_max": 22,
        }
    ]
    svc.search.return_value = {
        "query": "тест",
        "config": {
            "candidate_top_k": 30,
            "final_top_k": 5,
            "use_reranker": True,
            "context_selection": "anchor_page",
            "filters": {},
        },
        "results": [
            {
                "rank": 1,
                "chunk_id": "chunk-001",
                "document_id": "kr155_2",
                "document_title": "КР155",
                "section_id": "2",
                "section_title": "Диагностика",
                "label": "diagnosis",
                "specialty": "cardiology",
                "page_start": 12,
                "page_end": 13,
                "source": "docs/kr155_2.pdf",
                "text": "Диагноз ИБС.",
                "embedding_text": None,
                "scores": {
                    "dense_score": 0.95,
                    "reranker_score": 3.5,
                    "final_score": 3.5,
                },
                "context_source": "anchor",
            }
        ],
        "timing": {
            "query_embedding_time_ms": 100.0,
            "dense_retrieval_time_ms": 2.0,
            "reranking_time_ms": 500.0,
            "context_selection_time_ms": 3.0,
            "total_time_ms": 610.0,
        },
        "trace_id": "test-trace-uuid",
    }
    svc.batch_search.return_value = (
        [svc.search.return_value],
        650.0,
        650.0,
    )
    return svc
