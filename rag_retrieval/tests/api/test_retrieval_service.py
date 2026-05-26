from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api.retrieval_service import RetrievalService
from src.retrieval import RetrievalHit

from .conftest import SAMPLE_METADATA, make_hit

def _make_unloaded_service() -> RetrievalService:
    cfg = {
        "retrieval": {
            "embedding_model_key": "e5_large",
            "embedding_config": "configs/embedding_models.yaml",
            "embeddings_dir": "outputs/embeddings/e5_large",
            "reranker_key": "bge_reranker_v2_m3",
            "reranker_config": "configs/rerankers.yaml",
            "device": "cpu",
        },
        "performance": {"warmup_on_startup": False, "text_max_chars": 2000},
        "response": {"text_max_chars": 2000},
        "logging": {},
    }
    return RetrievalService(cfg)


def _make_loaded_service(
    metadata: List[Dict[str, Any]],
    embeddings: np.ndarray,
    mock_embedding_model,
    mock_reranker,
    mock_retriever,
) -> RetrievalService:
    svc = _make_unloaded_service()
    svc._metadata = metadata
    svc._embeddings = embeddings
    svc._run_info = {"model_key": "e5_large", "normalize": True, "embedding_dim": 1024}
    for i, m in enumerate(metadata):
        cid = m.get("id")
        if cid:
            svc._chunk_id_to_index[str(cid)] = i
    svc._retriever = mock_retriever
    svc._embedding_model = mock_embedding_model
    svc._reranker = mock_reranker
    svc._device = "cpu"
    svc._loaded = True
    svc._precompute_caches()
    return svc

class TestApplyFilters:

    def setup_method(self):
        self.svc = _make_unloaded_service()
        self.hits = [
            make_hit(1, "chunk-001", document_id="kr155_2", label="diagnosis", specialty="cardiology"),
            make_hit(2, "chunk-002", document_id="kr155_2", label="treatment", specialty="cardiology"),
            make_hit(3, "chunk-003", document_id="kr200_1", label="general", specialty="therapy"),
        ]

    def test_no_filters(self):
        result = self.svc._apply_filters(self.hits, {})
        assert result == self.hits

    def test_filter_by_document_id(self):
        result = self.svc._apply_filters(self.hits, {"document_id": "kr155_2"})
        assert len(result) == 2
        assert all(h.document_id == "kr155_2" for h in result)

    def test_filter_by_label(self):
        result = self.svc._apply_filters(self.hits, {"label": "diagnosis"})
        assert len(result) == 1
        assert result[0].chunk_id == "chunk-001"

    def test_filter_by_specialty(self):
        result = self.svc._apply_filters(self.hits, {"specialty": "therapy"})
        assert len(result) == 1
        assert result[0].chunk_id == "chunk-003"

    def test_combined_filters(self):
        result = self.svc._apply_filters(
            self.hits, {"document_id": "kr155_2", "label": "treatment"}
        )
        assert len(result) == 1
        assert result[0].chunk_id == "chunk-002"

    def test_filter_no_match_returns_empty(self):
        result = self.svc._apply_filters(self.hits, {"document_id": "nonexistent"})
        assert result == []

    def test_none_values_ignored(self):
        result = self.svc._apply_filters(
            self.hits, {"document_id": None, "label": None, "specialty": None}
        )
        assert result == self.hits

class TestComputeStats:

    def setup_method(self):
        self.svc = _make_unloaded_service()
        self.svc._metadata = SAMPLE_METADATA

        mock_retriever = MagicMock()
        mock_retriever.num_chunks = 3
        mock_retriever.embedding_dim = 1024
        self.svc._retriever = mock_retriever

    def test_num_chunks(self):
        stats = self.svc._compute_stats()
        assert stats["num_chunks"] == 3

    def test_num_documents(self):
        stats = self.svc._compute_stats()
        assert stats["num_documents"] == 2  

    def test_labels_count(self):
        stats = self.svc._compute_stats()
        assert stats["labels"]["diagnosis"] == 1
        assert stats["labels"]["treatment"] == 1
        assert stats["labels"]["general"] == 1

    def test_specialties_count(self):
        stats = self.svc._compute_stats()
        assert stats["specialties"]["cardiology"] == 2
        assert stats["specialties"]["therapy"] == 1

    def test_embedding_dim(self):
        stats = self.svc._compute_stats()
        assert stats["embedding_dim"] == 1024

    def test_empty_metadata(self):
        self.svc._metadata = []
        stats = self.svc._compute_stats()
        assert stats["num_chunks"] == 0
        assert stats["num_documents"] == 0
        assert stats["labels"] == {}


class TestComputeDocuments:

    def setup_method(self):
        self.svc = _make_unloaded_service()
        self.svc._metadata = SAMPLE_METADATA

    def test_document_count(self):
        docs = self.svc._compute_documents()
        assert len(docs) == 2

    def test_document_ids_sorted(self):
        docs = self.svc._compute_documents()
        ids = [d["document_id"] for d in docs]
        assert ids == sorted(ids)

    def test_num_chunks_per_document(self):
        docs = self.svc._compute_documents()
        doc_map = {d["document_id"]: d for d in docs}
        assert doc_map["kr155_2"]["num_chunks"] == 2
        assert doc_map["kr200_1"]["num_chunks"] == 1

    def test_page_min_max(self):
        docs = self.svc._compute_documents()
        doc_map = {d["document_id"]: d for d in docs}
        assert doc_map["kr155_2"]["page_min"] == 12
        assert doc_map["kr155_2"]["page_max"] == 22

    def test_labels_aggregated(self):
        docs = self.svc._compute_documents()
        doc_map = {d["document_id"]: d for d in docs}
        labels_155 = set(doc_map["kr155_2"]["labels"])
        assert "diagnosis" in labels_155
        assert "treatment" in labels_155

    def test_empty_metadata(self):
        self.svc._metadata = []
        docs = self.svc._compute_documents()
        assert docs == []


class TestHealthReady:

    def setup_method(self):
        self.svc = _make_unloaded_service()
        mock_retriever = MagicMock()
        mock_retriever.num_chunks = 3702
        self.svc._retriever = mock_retriever
        self.svc._embeddings = np.zeros((3702, 1024), dtype=np.float32)
        self.svc._metadata = SAMPLE_METADATA
        self.svc._embedding_model = MagicMock()
        self.svc._reranker = MagicMock()
        self.svc._device = "cpu"
        self.svc._loaded = True

    def test_health_ok(self):
        h = self.svc.get_health()
        assert h["status"] == "ok"
        assert h["loaded"] is True
        assert h["num_chunks"] == 3702
        assert h["device"] == "cpu"
        assert h["uptime_sec"] >= 0

    def test_health_not_loaded(self):
        self.svc._loaded = False
        h = self.svc.get_health()
        assert h["loaded"] is False

    def test_ready_all_true(self):
        r = self.svc.get_ready()
        assert r["ready"] is True
        assert all(r["checks"].values())

    def test_ready_no_reranker(self):
        self.svc._reranker = None
        r = self.svc.get_ready()
        assert r["ready"] is False
        assert r["checks"]["reranker_loaded"] is False

    def test_ready_no_embedding_model(self):
        self.svc._embedding_model = None
        r = self.svc.get_ready()
        assert r["ready"] is False
        assert r["checks"]["embedding_model_loaded"] is False

    def test_ready_no_embeddings(self):
        self.svc._embeddings = None
        r = self.svc.get_ready()
        assert r["ready"] is False
        assert r["checks"]["embeddings_loaded"] is False

class TestSearch:

    def setup_method(self):
        self.metadata = SAMPLE_METADATA

        rng = np.random.default_rng(0)
        self.embeddings = rng.random((3, 1024), dtype=np.float32)

        self.mock_emb_model = MagicMock()
        q_vec = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
        self.mock_emb_model.encode_queries.return_value = np.array([q_vec])

        self.mock_retriever = MagicMock()
        self.mock_retriever.embedding_dim = 1024
        self.mock_retriever.num_chunks = 3
        self.mock_retriever.search.return_value = [
            make_hit(1, "chunk-001", score=0.95),
            make_hit(2, "chunk-002", score=0.80, label="treatment"),
            make_hit(3, "chunk-003", score=0.60, document_id="kr200_1"),
        ]

        self.mock_reranker = MagicMock()
        self.mock_reranker.score.return_value = [3.5, 2.1, 1.0]

        self.svc = _make_loaded_service(
            metadata=self.metadata,
            embeddings=self.embeddings,
            mock_embedding_model=self.mock_emb_model,
            mock_reranker=self.mock_reranker,
            mock_retriever=self.mock_retriever,
        )

    def test_search_calls_encode_queries(self):
        self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=3,
            use_reranker=True,
            context_selection="none",
            filters=None,
            include_text=True,
            include_embedding_text=False,
        )
        self.mock_emb_model.encode_queries.assert_called_once()
        call_args = self.mock_emb_model.encode_queries.call_args
        assert call_args[0][0] == ["тест"]

    def test_search_calls_retriever(self):
        self.svc.search(
            query="тест",
            candidate_top_k=15,
            final_top_k=3,
            use_reranker=True,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        self.mock_retriever.search.assert_called_once()
        _, kwargs = self.mock_retriever.search.call_args
        assert kwargs.get("top_k") == 15 or self.mock_retriever.search.call_args[0][1] == 15

    def test_search_with_reranker_calls_score(self):
        self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=2,
            use_reranker=True,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        self.mock_reranker.score.assert_called_once()

    def test_search_without_reranker_skips_score(self):
        self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=2,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        self.mock_reranker.score.assert_not_called()

    def test_search_returns_correct_structure(self):
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=2,
            use_reranker=True,
            context_selection="none",
            filters=None,
            include_text=True,
            include_embedding_text=False,
        )
        assert "query" in result
        assert "results" in result
        assert "timing" in result
        assert "trace_id" in result
        assert "config" in result

    def test_search_timing_keys(self):
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=2,
            use_reranker=True,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        timing = result["timing"]
        for key in (
            "query_embedding_time_ms",
            "dense_retrieval_time_ms",
            "reranking_time_ms",
            "context_selection_time_ms",
            "total_time_ms",
        ):
            assert key in timing, f"Missing key: {key}"
            assert timing[key] >= 0

    def test_search_trace_id_is_uuid_string(self):
        import uuid
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=2,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        # Проверяем формат UUID4
        uuid.UUID(result["trace_id"], version=4)

    def test_search_include_text_false(self):
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=3,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        for r in result["results"]:
            assert r["text"] is None

    def test_search_include_text_true(self):
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=1,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=True,
            include_embedding_text=False,
        )
        for r in result["results"]:
            assert r["text"] is not None

    def test_search_text_truncated_to_max_chars(self):
        long_text = "А" * 5000
        self.mock_retriever.search.return_value = [
            make_hit(1, "chunk-001", score=0.9, text=long_text),
        ]
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=1,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=True,
            include_embedding_text=False,
        )
        assert len(result["results"][0]["text"]) <= 2000

    def test_search_include_embedding_text(self):
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=1,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=True,
        )
        for r in result["results"]:
            if r["chunk_id"] == "chunk-001":
                assert r["embedding_text"] is not None

    def test_search_with_document_filter(self):
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=5,
            use_reranker=False,
            context_selection="none",
            filters={"document_id": "kr200_1"},
            include_text=False,
            include_embedding_text=False,
        )
        for r in result["results"]:
            assert r["document_id"] == "kr200_1"

    def test_search_scores_structure(self):
        result = self.svc.search(
            query="тест",
            candidate_top_k=10,
            final_top_k=2,
            use_reranker=True,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        for r in result["results"]:
            scores = r["scores"]
            assert "dense_score" in scores
            assert "reranker_score" in scores
            assert "final_score" in scores

    def test_search_raises_if_reranker_none_and_use_reranker(self):
        self.svc._reranker = None
        with pytest.raises(RuntimeError, match="reranker is not loaded"):
            self.svc.search(
                query="тест",
                candidate_top_k=10,
                final_top_k=2,
                use_reranker=True,
                context_selection="none",
                filters=None,
                include_text=False,
                include_embedding_text=False,
            )

    def test_search_not_loaded_raises(self):
        self.svc._loaded = False
        with pytest.raises(RuntimeError, match="not loaded"):
            self.svc.search(
                query="тест",
                candidate_top_k=10,
                final_top_k=2,
                use_reranker=False,
                context_selection="none",
                filters=None,
                include_text=False,
                include_embedding_text=False,
            )

class TestBatchSearch:

    def setup_method(self):
        self.svc = _make_unloaded_service()
        self.svc._loaded = True
        # Заменяем search мок-версией для изоляции
        self.svc.search = MagicMock(return_value={
            "query": "q",
            "config": {},
            "results": [{"rank": 1, "chunk_id": "x"}],
            "timing": {"total_time_ms": 500.0},
            "trace_id": "abc",
        })

    def test_batch_calls_search_for_each_query(self):
        queries = ["q1", "q2", "q3"]
        self.svc.batch_search(
            queries=queries,
            candidate_top_k=30,
            final_top_k=5,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        assert self.svc.search.call_count == 3

    def test_batch_returns_tuple_with_timing(self):
        items, total_ms, avg_ms = self.svc.batch_search(
            queries=["q1", "q2"],
            candidate_top_k=10,
            final_top_k=3,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        assert len(items) == 2
        assert total_ms >= 0
        assert avg_ms >= 0

    def test_batch_single_query_avg_equals_total(self):
        items, total_ms, avg_ms = self.svc.batch_search(
            queries=["q1"],
            candidate_top_k=10,
            final_top_k=3,
            use_reranker=False,
            context_selection="none",
            filters=None,
            include_text=False,
            include_embedding_text=False,
        )
        assert items[0]["query"] == "q"
        assert abs(total_ms - avg_ms) < 0.1

class TestDoRerank:

    def setup_method(self):
        self.svc = _make_unloaded_service()
        self.svc._metadata = SAMPLE_METADATA
        for i, m in enumerate(SAMPLE_METADATA):
            cid = m.get("id")
            if cid:
                self.svc._chunk_id_to_index[str(cid)] = i

        self.mock_reranker = MagicMock()
        self.svc._reranker = self.mock_reranker

    def test_rerank_returns_sorted_by_score(self):
        self.mock_reranker.score.return_value = [1.0, 3.0, 2.0]
        hits = [
            make_hit(1, "chunk-001", score=0.9),
            make_hit(2, "chunk-002", score=0.8),
            make_hit(3, "chunk-003", score=0.7),
        ]
        reranked, scores, dense_ranks = self.svc._do_rerank("тест", hits)
        assert scores[0] == 3.0
        assert scores[1] == 2.0
        assert scores[2] == 1.0

    def test_rerank_dense_scores_preserved(self):
        self.mock_reranker.score.return_value = [1.0, 3.0, 2.0]
        hits = [
            make_hit(1, "chunk-001", score=0.9),
            make_hit(2, "chunk-002", score=0.8),
            make_hit(3, "chunk-003", score=0.7),
        ]
        reranked, scores, dense_ranks = self.svc._do_rerank("тест", hits)
        dense_score_values = {h.chunk_id: h.score for h in reranked}
        assert abs(dense_score_values["chunk-001"] - 0.9) < 1e-5
        assert abs(dense_score_values["chunk-002"] - 0.8) < 1e-5

    def test_rerank_dense_ranks_recorded(self):
        self.mock_reranker.score.return_value = [1.0, 3.0, 2.0]
        hits = [
            make_hit(1, "chunk-001", score=0.9),
            make_hit(2, "chunk-002", score=0.8),
            make_hit(3, "chunk-003", score=0.7),
        ]
        reranked, scores, dense_ranks = self.svc._do_rerank("тест", hits)
        assert dense_ranks[0] == 2

    def test_rerank_empty_hits(self):
        result = self.svc._do_rerank("тест", [])
        assert result == ([], [], [])
        self.mock_reranker.score.assert_not_called()

    def test_rerank_new_ranks_sequential(self):
        self.mock_reranker.score.return_value = [3.0, 1.0, 2.0]
        hits = [
            make_hit(1, "chunk-001"),
            make_hit(2, "chunk-002"),
            make_hit(3, "chunk-003"),
        ]
        reranked, _, _ = self.svc._do_rerank("тест", hits)
        ranks = [h.rank for h in reranked]
        assert ranks == [1, 2, 3]
