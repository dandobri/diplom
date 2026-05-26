"""Этап 5. Retrieval: numpy-поиск ближайших чанков.

Дизайн: класс Retriever инкапсулирует индекс. Сейчас реализован
NumpyRetriever (dot product / cosine на нормализованных векторах).
Для замены на FAISS достаточно реализовать тот же интерфейс
(Retriever.search(query_vec, top_k) -> List[RetrievalHit]).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    

    rank: int
    score: float
    chunk_id: str
    document_id: Optional[str]
    document_title: Optional[str]
    section_id: Optional[str]
    section_title: Optional[str]
    label: Optional[str]
    source: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, text_preview_chars: Optional[int] = None) -> Dict[str, Any]:
        
        d = {
            "rank": self.rank,
            "score": float(self.score),
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "label": self.label,
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
        if text_preview_chars is not None:
            d["text_preview"] = (self.text or "")[:text_preview_chars]
        else:
            d["text"] = self.text
        return d


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Retriever:
    

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[RetrievalHit]:
        raise NotImplementedError


class NumpyRetriever(Retriever):
    """Numpy-поиск через dot product.

    Если embeddings уже нормализованы, dot == cosine similarity.
    Если нет — конструктор может нормализовать их.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: Sequence[Dict[str, Any]],
        already_normalized: bool = True,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"embeddings rows ({embeddings.shape[0]}) != metadata rows ({len(metadata)})"
            )
        emb = np.asarray(embeddings, dtype=np.float32)
        if not already_normalized:
            emb = _l2_normalize(emb)
        self._emb = emb
        self._meta: List[Dict[str, Any]] = list(metadata)
        self._already_normalized = already_normalized

    @property
    def num_chunks(self) -> int:
        return self._emb.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self._emb.shape[1]

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[RetrievalHit]:
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query dim {q.shape[0]} != index dim {self.embedding_dim}"
            )
        if not self._already_normalized:
            n = float(np.linalg.norm(q))
            if n > 0:
                q = q / n

        scores = self._emb @ q  
        k = min(top_k, scores.shape[0])
        if k == 0:
            return []
        
        idx_part = np.argpartition(-scores, k - 1)[:k]
        order = idx_part[np.argsort(-scores[idx_part])]

        hits: List[RetrievalHit] = []
        for rank, i in enumerate(order, start=1):
            meta = self._meta[int(i)]
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=float(scores[int(i)]),
                    chunk_id=str(meta.get("id", "")),
                    document_id=meta.get("document_id"),
                    document_title=meta.get("document_title"),
                    section_id=meta.get("section_id"),
                    section_title=meta.get("section_title"),
                    label=meta.get("label"),
                    source=meta.get("source"),
                    page_start=meta.get("page_start"),
                    page_end=meta.get("page_end"),
                    text=meta.get("text", "") or "",
                    extra={
                        k: meta.get(k)
                        for k in ("specialty", "stage", "content_hash", "chunk_index")
                        if k in meta
                    },
                )
            )
        return hits


def build_retriever(
    embeddings: np.ndarray,
    metadata: Sequence[Dict[str, Any]],
    already_normalized: bool = True,
    backend: str = "numpy",
) -> Retriever:
    
    backend = backend.lower()
    if backend == "numpy":
        return NumpyRetriever(embeddings, metadata, already_normalized=already_normalized)
    if backend == "faiss":
        raise NotImplementedError(
            "FAISS backend is not implemented yet; use 'numpy' until corpus > 100k."
        )
    raise ValueError(f"Unknown retriever backend: {backend}")
