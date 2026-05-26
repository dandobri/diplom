from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..context_selection import select_context_top_k
from ..embedding_models import EmbeddingModel, load_embedding_model
from ..io_utils import load_model_config, read_json, read_jsonl_list
from ..reranker import Reranker, build_reranker, format_passage_for_reranker, load_reranker_config
from ..retrieval import NumpyRetriever, RetrievalHit

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg = config
        self._rcfg: Dict[str, Any] = config.get("retrieval", {})
        self._perf_cfg: Dict[str, Any] = config.get("performance", {})
        self._resp_cfg: Dict[str, Any] = config.get("response", {})

        self._embeddings: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._run_info: Dict[str, Any] = {}
        self._chunk_id_to_index: Dict[str, int] = {}

        self._retriever: Optional[NumpyRetriever] = None
        self._embedding_model: Optional[EmbeddingModel] = None
        self._reranker: Optional[Reranker] = None

        self._device: str = "cpu"
        self._start_time: float = time.time()
        self._loaded: bool = False

        self._docs_cache: Optional[List[Dict[str, Any]]] = None
        self._stats_cache: Optional[Dict[str, Any]] = None

    def load(self) -> None:
        rcfg = self._rcfg
        device = rcfg.get("device", "auto")
        model_key = rcfg.get("embedding_model_key", "e5_large")
        embeddings_dir = Path(rcfg.get("embeddings_dir", "outputs/embeddings/e5_large"))
        embedding_config = rcfg.get("embedding_config", "configs/embedding_models.yaml")
        reranker_key = rcfg.get("reranker_key", "bge_reranker_v2_m3")
        reranker_config = rcfg.get("reranker_config", "configs/rerankers.yaml")

        logger.info("Loading embedding artifacts from %s", embeddings_dir)
        self._embeddings, self._metadata, self._run_info = self._load_artifacts(
            embeddings_dir, model_key
        )
        for i, m in enumerate(self._metadata):
            cid = m.get("id")
            if cid:
                self._chunk_id_to_index[str(cid)] = i

        already_normalized = bool(self._run_info.get("normalize", True))
        self._retriever = NumpyRetriever(
            self._embeddings, self._metadata, already_normalized=already_normalized
        )
        logger.info(
            "Retriever ready: %d chunks, dim=%d",
            self._retriever.num_chunks,
            self._retriever.embedding_dim,
        )

        logger.info("Loading embedding model: %s", model_key)
        emb_cfg = load_model_config(embedding_config, model_key)
        self._embedding_model = load_embedding_model(emb_cfg, device=device)
        _ = self._embedding_model.embedding_dim
        self._device = self._embedding_model.device
        logger.info("Embedding model loaded on device=%s", self._device)

        logger.info("Loading reranker: %s", reranker_key)
        rer_cfg = load_reranker_config(reranker_config, reranker_key)
        self._reranker = build_reranker(rer_cfg, device=device)
        try:
            _ = self._reranker.score("warmup", ["warmup passage"])
            logger.info("Reranker loaded")
        except Exception:
            logger.exception("Reranker warmup failed; reranker will be unavailable")
            self._reranker = None

        if self._perf_cfg.get("warmup_on_startup", True):
            self._warmup()

        self._loaded = True
        self._precompute_caches()
        logger.info("RetrievalService fully loaded and ready")

    def _load_artifacts(
        self, embeddings_dir: Path, model_key: str
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        emb_path = embeddings_dir / "embeddings.npy"
        meta_path = embeddings_dir / "metadata.jsonl"
        info_path = embeddings_dir / "run_info.json"
        for p in (emb_path, meta_path, info_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing artifact: {p}")
        embeddings = np.load(emb_path).astype(np.float32, copy=False)
        metadata = read_jsonl_list(meta_path)
        run_info = read_json(info_path)
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"embeddings rows ({embeddings.shape[0]}) != metadata rows ({len(metadata)})"
            )
        declared_key = run_info.get("model_key")
        if declared_key and declared_key != model_key:
            raise ValueError(
                f"run_info.model_key='{declared_key}' != configured key='{model_key}'"
            )
        return embeddings, metadata, run_info

    def _warmup(self) -> None:
        logger.info("Running pipeline warmup...")
        try:
            warmup_query = "Пациент с болью за грудиной"
            q_vec = self._embedding_model.encode_queries([warmup_query], show_progress_bar=False)[0]
            candidates = self._retriever.search(q_vec, top_k=5)
            if self._reranker and candidates:
                passages = [
                    format_passage_for_reranker(
                        self._metadata[self._chunk_id_to_index.get(h.chunk_id, 0)]
                        if self._chunk_id_to_index.get(h.chunk_id) is not None
                        else {}
                    )
                    for h in candidates
                ]
                self._reranker.score(warmup_query, passages)
            logger.info("Warmup complete")
        except Exception:
            logger.exception("Warmup failed (non-fatal)")

    def _precompute_caches(self) -> None:
        logger.info("Precomputing documents and stats caches...")
        self._docs_cache = self._compute_documents()
        self._stats_cache = self._compute_stats()
        logger.info(
            "Caches ready: %d documents, %d chunks",
            len(self._docs_cache),
            self._stats_cache["num_chunks"],
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_health(self) -> Dict[str, Any]:
        model_key = self._rcfg.get("embedding_model_key", "")
        reranker_key = self._rcfg.get("reranker_key", "")
        num_chunks = self._retriever.num_chunks if self._retriever else 0
        return {
            "status": "ok",
            "service": "rag-retrieval-api",
            "loaded": self._loaded,
            "embedding_model_key": model_key,
            "reranker_key": reranker_key,
            "num_chunks": num_chunks,
            "device": self._device,
            "uptime_sec": round(time.time() - self._start_time, 1),
        }

    def get_ready(self) -> Dict[str, Any]:
        embeddings_ok = self._embeddings is not None and len(self._embeddings) > 0
        metadata_ok = len(self._metadata) > 0
        model_ok = self._embedding_model is not None
        reranker_ok = self._reranker is not None
        checks = {
            "embeddings_loaded": embeddings_ok,
            "metadata_loaded": metadata_ok,
            "embedding_model_loaded": model_ok,
            "reranker_loaded": reranker_ok,
        }
        ready = all(checks.values())
        return {"ready": ready, "checks": checks}

    def search(
        self,
        query: str,
        candidate_top_k: int,
        final_top_k: int,
        use_reranker: bool,
        context_selection: str,
        filters: Optional[Dict[str, Any]],
        include_text: bool,
        include_embedding_text: bool,
    ) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Service is not loaded yet")

        trace_id = str(uuid.uuid4())
        text_max = self._resp_cfg.get("text_max_chars", 2000)

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        q_vec = self._embedding_model.encode_queries([query], show_progress_bar=False)[0]
        query_embedding_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        candidates = self._retriever.search(q_vec, top_k=candidate_top_k)
        dense_retrieval_ms = (time.perf_counter() - t1) * 1000.0

        if filters:
            candidates = self._apply_filters(candidates, filters)

        t2 = time.perf_counter()
        if use_reranker:
            if self._reranker is None:
                raise RuntimeError(
                    "use_reranker=true but reranker is not loaded. "
                    "Check service readiness at /ready."
                )
            reranked, rer_scores, dense_ranks = self._do_rerank(query, candidates)
        else:
            reranked = candidates
            rer_scores = [float(h.score) for h in candidates]
            dense_ranks = [h.rank for h in candidates]
        reranking_ms = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        final_hits, final_scores, _ = select_context_top_k(
            reranked=reranked,
            reranker_scores=rer_scores,
            dense_ranks_in_rerank=dense_ranks,
            metadata=self._metadata,
            mode=context_selection,
            final_top_k=final_top_k,
        )
        context_selection_ms = (time.perf_counter() - t3) * 1000.0

        total_ms = (time.perf_counter() - t_total) * 1000.0

        results = []
        for hit, fscore in zip(final_hits, final_scores):
            dense_score = float(hit.score)
            reranker_score = float(fscore) if fscore is not None else None
            final_score = reranker_score if reranker_score is not None else dense_score

            text: Optional[str] = None
            if include_text:
                raw = hit.text or ""
                text = raw[:text_max] if raw else raw

            emb_text: Optional[str] = None
            if include_embedding_text:
                meta_idx = self._chunk_id_to_index.get(hit.chunk_id)
                if meta_idx is not None:
                    raw_et = self._metadata[meta_idx].get("embedding_text") or ""
                    emb_text = raw_et[:text_max] if raw_et else None

            extra = hit.extra if isinstance(hit.extra, dict) else {}
            results.append(
                {
                    "rank": hit.rank,
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "document_title": hit.document_title,
                    "section_id": hit.section_id,
                    "section_title": hit.section_title,
                    "label": hit.label,
                    "specialty": extra.get("specialty"),
                    "page_start": hit.page_start,
                    "page_end": hit.page_end,
                    "source": hit.source,
                    "text": text,
                    "embedding_text": emb_text,
                    "scores": {
                        "dense_score": dense_score,
                        "reranker_score": reranker_score,
                        "final_score": final_score,
                    },
                    "context_source": extra.get("context_source"),
                }
            )

        return {
            "query": query,
            "config": {
                "candidate_top_k": candidate_top_k,
                "final_top_k": final_top_k,
                "use_reranker": use_reranker,
                "context_selection": context_selection,
                "filters": (
                    {k: v for k, v in (filters or {}).items() if v is not None}
                ),
            },
            "results": results,
            "timing": {
                "query_embedding_time_ms": round(query_embedding_ms, 2),
                "dense_retrieval_time_ms": round(dense_retrieval_ms, 2),
                "reranking_time_ms": round(reranking_ms, 2),
                "context_selection_time_ms": round(context_selection_ms, 2),
                "total_time_ms": round(total_ms, 2),
            },
            "trace_id": trace_id,
        }

    def batch_search(
        self,
        queries: List[str],
        candidate_top_k: int,
        final_top_k: int,
        use_reranker: bool,
        context_selection: str,
        filters: Optional[Dict[str, Any]],
        include_text: bool,
        include_embedding_text: bool,
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        t_start = time.perf_counter()
        results = []
        for q in queries:
            r = self.search(
                query=q,
                candidate_top_k=candidate_top_k,
                final_top_k=final_top_k,
                use_reranker=use_reranker,
                context_selection=context_selection,
                filters=filters,
                include_text=include_text,
                include_embedding_text=include_embedding_text,
            )
            results.append(r)
        total_ms = (time.perf_counter() - t_start) * 1000.0
        avg_ms = total_ms / len(queries) if queries else 0.0
        return results, round(total_ms, 2), round(avg_ms, 2)

    def stats(self) -> Dict[str, Any]:
        if self._stats_cache is not None:
            return self._stats_cache
        return self._compute_stats()

    def documents(self) -> List[Dict[str, Any]]:
        if self._docs_cache is not None:
            return self._docs_cache
        return self._compute_documents()

    def _compute_stats(self) -> Dict[str, Any]:
        labels: Counter = Counter()
        specialties: Counter = Counter()
        doc_ids: set = set()
        for m in self._metadata:
            if m.get("label"):
                labels[m["label"]] += 1
            if m.get("specialty"):
                specialties[m["specialty"]] += 1
            if m.get("document_id"):
                doc_ids.add(m["document_id"])
        return {
            "num_chunks": len(self._metadata),
            "num_documents": len(doc_ids),
            "labels": dict(labels),
            "specialties": dict(specialties),
            "embedding_dim": self._retriever.embedding_dim if self._retriever else None,
        }

    def _compute_documents(self) -> List[Dict[str, Any]]:
        doc_data: Dict[str, Dict[str, Any]] = {}
        for m in self._metadata:
            doc_id = m.get("document_id")
            if not doc_id:
                continue
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "document_id": doc_id,
                    "document_title": m.get("document_title"),
                    "num_chunks": 0,
                    "labels": set(),
                    "page_min": None,
                    "page_max": None,
                }
            d = doc_data[doc_id]
            d["num_chunks"] += 1
            if m.get("label"):
                d["labels"].add(m["label"])
            ps = m.get("page_start")
            pe = m.get("page_end")
            if ps is not None:
                d["page_min"] = min(d["page_min"], ps) if d["page_min"] is not None else ps
            if pe is not None:
                d["page_max"] = max(d["page_max"], pe) if d["page_max"] is not None else pe

        docs = []
        for doc_id in sorted(doc_data.keys()):
            d = doc_data[doc_id]
            docs.append(
                {
                    "document_id": d["document_id"],
                    "document_title": d["document_title"],
                    "num_chunks": d["num_chunks"],
                    "labels": sorted(d["labels"]),
                    "page_min": d["page_min"],
                    "page_max": d["page_max"],
                }
            )
        return docs

    def _apply_filters(
        self, hits: List[RetrievalHit], filters: Dict[str, Any]
    ) -> List[RetrievalHit]:
        doc_f = filters.get("document_id")
        label_f = filters.get("label")
        spec_f = filters.get("specialty")
        if not any([doc_f, label_f, spec_f]):
            return hits
        filtered = []
        for h in hits:
            if doc_f and h.document_id != doc_f:
                continue
            if label_f and h.label != label_f:
                continue
            if spec_f:
                extra = h.extra if isinstance(h.extra, dict) else {}
                if extra.get("specialty") != spec_f:
                    continue
            filtered.append(h)
        return filtered

    def _do_rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
    ) -> Tuple[List[RetrievalHit], List[float], List[int]]:
        if not hits:
            return [], [], []
        passages = []
        for h in hits:
            meta_idx = self._chunk_id_to_index.get(h.chunk_id)
            meta = self._metadata[meta_idx] if meta_idx is not None else {}
            passages.append(format_passage_for_reranker(meta))

        scores = self._reranker.score(query, passages)
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)

        reranked: List[RetrievalHit] = []
        reranked_scores: List[float] = []
        dense_ranks: List[int] = []
        for new_rank, idx in enumerate(order, start=1):
            original = hits[idx]
            new_hit = RetrievalHit(
                rank=new_rank,
                score=original.score,
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                document_title=original.document_title,
                section_id=original.section_id,
                section_title=original.section_title,
                label=original.label,
                source=original.source,
                page_start=original.page_start,
                page_end=original.page_end,
                text=original.text,
                extra=dict(original.extra) if isinstance(original.extra, dict) else {},
            )
            reranked.append(new_hit)
            reranked_scores.append(float(scores[idx]))
            dense_ranks.append(original.rank)
        return reranked, reranked_scores, dense_ranks
