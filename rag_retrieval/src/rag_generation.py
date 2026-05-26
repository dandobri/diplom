from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .context_selection import SUPPORTED_MODES, select_context_top_k
from .embedding_models import load_embedding_model
from .io_utils import load_model_config, read_json, read_jsonl_list
from .llm_client import LLMClient
from .prompt_templates import build_no_rag_messages, build_rag_messages
from .reranker import (
    Reranker,
    build_reranker,
    format_passage_for_reranker,
    load_reranker_config,
)
from .retrieval import NumpyRetriever, RetrievalHit

logger = logging.getLogger(__name__)




def load_retrieval_artifacts(
    embeddings_dir: str,
    embedding_model_key: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    
    d = Path(embeddings_dir)
    emb_path = d / "embeddings.npy"
    meta_path = d / "metadata.jsonl"
    info_path = d / "run_info.json"
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
    if int(run_info.get("embedding_dim", -1)) != int(embeddings.shape[1]):
        raise ValueError(
            f"embedding_dim mismatch: run_info={run_info.get('embedding_dim')} "
            f"actual={embeddings.shape[1]}"
        )
    if run_info.get("model_key") != embedding_model_key:
        raise ValueError(
            f"run_info.model_key={run_info.get('model_key')!r} != "
            f"--embedding-model-key={embedding_model_key!r}"
        )
    return embeddings, metadata, run_info




@dataclass
class RetrievalEngine:
    """Инкапсулирует все артефакты retrieval-пайплайна.

    Создается один раз через `from_config(...)` и переиспользуется для всех
    запросов в скрипте — embedding model и reranker не перегружаются.
    """

    embedding_model: Any
    reranker: Reranker
    retriever: NumpyRetriever
    metadata: List[Dict[str, Any]]
    chunk_id_to_index: Dict[str, int]
    run_info: Dict[str, Any]
    candidate_top_k: int = 30
    final_top_k: int = 5
    context_selection: str = "anchor_page"
    context_page_tolerance: int = 1

    @classmethod
    def from_config(
        cls,
        *,
        embeddings_dir: str,
        embedding_model_key: str,
        embedding_config: str,
        reranker_key: str,
        reranker_config: str,
        candidate_top_k: int = 30,
        final_top_k: int = 5,
        context_selection: str = "anchor_page",
        context_page_tolerance: int = 1,
        device: str = "auto",
    ) -> "RetrievalEngine":
        if context_selection not in SUPPORTED_MODES:
            raise ValueError(
                f"context_selection={context_selection!r} not in {SUPPORTED_MODES}"
            )

        embeddings, metadata, run_info = load_retrieval_artifacts(
            embeddings_dir, embedding_model_key
        )

        chunk_id_to_index: Dict[str, int] = {}
        for i, m in enumerate(metadata):
            cid = m.get("id")
            if cid:
                chunk_id_to_index[str(cid)] = i

        retriever = NumpyRetriever(
            embeddings,
            metadata,
            already_normalized=bool(run_info.get("normalize", True)),
        )

        emb_cfg = load_model_config(embedding_config, embedding_model_key)
        embedding_model = load_embedding_model(emb_cfg, device=device)

        rer_cfg = load_reranker_config(reranker_config, reranker_key)
        reranker = build_reranker(rer_cfg, device=device)

        logger.info(
            "RetrievalEngine ready: docs=%d dim=%d emb=%s rer=%s mode=%s top_k=%d",
            embeddings.shape[0],
            embeddings.shape[1],
            embedding_model_key,
            reranker_key,
            context_selection,
            final_top_k,
        )
        return cls(
            embedding_model=embedding_model,
            reranker=reranker,
            retriever=retriever,
            metadata=metadata,
            chunk_id_to_index=chunk_id_to_index,
            run_info=run_info,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
            context_selection=context_selection,
            context_page_tolerance=context_page_tolerance,
        )

    

    def retrieve(self, query: str) -> Tuple[
        List[RetrievalHit], List[Optional[float]], List[Optional[int]], Dict[str, float]
    ]:
        
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        q_vec = self.embedding_model.encode_queries([query], show_progress_bar=False)[0]
        timings["query_embedding_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        candidates = self.retriever.search(q_vec, top_k=self.candidate_top_k)
        timings["dense_retrieval_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        reranked, rer_scores, dense_ranks = self._rerank(candidates, query)
        timings["reranking_ms"] = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        final_hits, final_scores, final_dranks = select_context_top_k(
            reranked=reranked,
            reranker_scores=rer_scores,
            dense_ranks_in_rerank=dense_ranks,
            metadata=self.metadata,
            mode=self.context_selection,
            final_top_k=self.final_top_k,
            page_tolerance=self.context_page_tolerance,
        )
        timings["context_selection_ms"] = (time.perf_counter() - t3) * 1000.0
        timings["total_retrieval_ms"] = sum(timings.values())
        return final_hits, final_scores, final_dranks, timings

    def _rerank(
        self,
        hits: Sequence[RetrievalHit],
        query: str,
    ) -> Tuple[List[RetrievalHit], List[float], List[int]]:
        if not hits:
            return [], [], []
        passages: List[str] = []
        for h in hits:
            idx = self.chunk_id_to_index.get(h.chunk_id)
            meta = self.metadata[idx] if idx is not None else {}
            passages.append(format_passage_for_reranker(meta))
        scores = self.reranker.score(query, passages)
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        reranked: List[RetrievalHit] = []
        reranked_scores: List[float] = []
        dense_ranks: List[int] = []
        for new_rank, idx in enumerate(order, start=1):
            orig = hits[idx]
            reranked.append(
                RetrievalHit(
                    rank=new_rank,
                    score=orig.score,
                    chunk_id=orig.chunk_id,
                    document_id=orig.document_id,
                    document_title=orig.document_title,
                    section_id=orig.section_id,
                    section_title=orig.section_title,
                    label=orig.label,
                    source=orig.source,
                    page_start=orig.page_start,
                    page_end=orig.page_end,
                    text=orig.text,
                    extra=dict(orig.extra) if isinstance(orig.extra, dict) else {},
                )
            )
            reranked_scores.append(scores[idx])
            dense_ranks.append(orig.rank)
        return reranked, reranked_scores, dense_ranks




def format_retrieved_context(
    hits: Sequence[RetrievalHit],
    *,
    text_chars_per_chunk: int = 1500,
) -> str:
    
    blocks: List[str] = []
    for i, h in enumerate(hits, start=1):
        page_str = ""
        if h.page_start is not None and h.page_end is not None:
            page_str = f"{h.page_start}-{h.page_end}"
        elif h.page_start is not None:
            page_str = str(h.page_start)
        text = (h.text or "").strip()
        if len(text) > text_chars_per_chunk:
            text = text[:text_chars_per_chunk] + "…"
        block = (
            f"[Источник S{i}]\n"
            f"chunk_id: {h.chunk_id or ''}\n"
            f"document_id: {h.document_id or ''}\n"
            f"document_title: {h.document_title or ''}\n"
            f"section_title: {h.section_title or ''}\n"
            f"label: {h.label or ''}\n"
            f"pages: {page_str}\n"
            f"text:\n{text}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def hit_to_chunk_record(
    hit: RetrievalHit,
    source_id: str,
    *,
    reranker_score: Optional[float],
    dense_rank: Optional[int],
) -> Dict[str, Any]:
    extra = hit.extra if isinstance(hit.extra, dict) else {}
    return {
        "source_id": source_id,
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "document_title": hit.document_title,
        "section_id": hit.section_id,
        "section_title": hit.section_title,
        "label": hit.label,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "text": hit.text,
        "dense_score": float(hit.score) if hit.score is not None else None,
        "dense_rank": dense_rank,
        "reranker_score": float(reranker_score) if reranker_score is not None else None,
        "context_source": extra.get("context_source"),
    }




_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_llm_json(text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Извлекает и парсит JSON из ответа LLM.

    Возвращает (json_dict_or_None, errors). Поддерживает три формата:
        1. чистый JSON;
        2. ```json ... ``` markdown-блок;
        3. JSON с произвольным окружающим текстом — берём первую `{...}` пару.
    """
    errors: List[str] = []
    if not text or not isinstance(text, str):
        return None, ["empty_response"]

    stripped = text.strip()
    
    try:
        return json.loads(stripped), errors
    except json.JSONDecodeError as e:
        errors.append(f"direct_parse_failed: {e.msg}")

    
    for match in _JSON_BLOCK_RE.finditer(stripped):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate), errors
        except json.JSONDecodeError as e:
            errors.append(f"code_block_parse_failed: {e.msg}")

    
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate), errors
        except json.JSONDecodeError as e:
            errors.append(f"braces_parse_failed: {e.msg}")

    errors.append("no_valid_json_found")
    return None, errors




def _iter_citations(answer: Dict[str, Any]):
    
    for sect in ("differential_diagnoses", "recommended_next_steps", "red_flags"):
        items = answer.get(sect) or []
        for i, item in enumerate(items):
            for c in item.get("citations") or []:
                if isinstance(c, dict):
                    yield sect, i, c


def validate_citations(
    answer_json: Optional[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Проверяет, что все citations указывают на реальные источники из retrieved.

    Считает грубую citation_coverage_estimate: доля diagnosis + recommendation
    items, у которых хотя бы одна citation.
    """
    if not isinstance(answer_json, dict):
        return {
            "valid": False,
            "errors": ["no_answer_json"],
            "citation_count": 0,
            "invalid_citation_count": 0,
            "citation_coverage_estimate": None,
        }

    valid_source_ids = {c.get("source_id") for c in retrieved_chunks if c.get("source_id")}
    valid_chunk_ids = {c.get("chunk_id") for c in retrieved_chunks if c.get("chunk_id")}
    valid_doc_ids = {c.get("document_id") for c in retrieved_chunks if c.get("document_id")}

    total = 0
    invalid = 0
    errors: List[str] = []

    for sect, idx, c in _iter_citations(answer_json):
        total += 1
        sid = c.get("source_id")
        cid = c.get("chunk_id")
        did = c.get("document_id")
        is_invalid = False
        if sid and sid not in valid_source_ids:
            errors.append(f"{sect}[{idx}].citation: unknown source_id={sid!r}")
            is_invalid = True
        if cid and cid not in valid_chunk_ids:
            errors.append(f"{sect}[{idx}].citation: unknown chunk_id={cid!r}")
            is_invalid = True
        if did and did not in valid_doc_ids:
            errors.append(f"{sect}[{idx}].citation: unknown document_id={did!r}")
            is_invalid = True
        if is_invalid:
            invalid += 1

    
    cov_items_total = 0
    cov_items_with_citations = 0
    for sect in ("differential_diagnoses", "recommended_next_steps"):
        items = answer_json.get(sect) or []
        for item in items:
            cov_items_total += 1
            if item.get("citations"):
                cov_items_with_citations += 1
    coverage = (
        cov_items_with_citations / cov_items_total if cov_items_total else None
    )

    return {
        "valid": invalid == 0,
        "errors": errors,
        "citation_count": total,
        "invalid_citation_count": invalid,
        "citation_coverage_estimate": (
            round(coverage, 4) if coverage is not None else None
        ),
    }




def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_rag_answer(
    *,
    case_id: str,
    patient_case: str,
    engine: RetrievalEngine,
    llm_client: LLMClient,
    text_chars_per_chunk: int = 1500,
    extra_meta: Optional[Dict[str, Any]] = None,
    prompt_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    t0 = time.perf_counter()
    errors: List[str] = []

    final_hits, rer_scores, dense_ranks, timings_ms = engine.retrieve(patient_case)
    retrieved_chunks: List[Dict[str, Any]] = []
    for i, (h, sc, dr) in enumerate(zip(final_hits, rer_scores, dense_ranks), start=1):
        retrieved_chunks.append(
            hit_to_chunk_record(h, source_id=f"S{i}", reranker_score=sc, dense_rank=dr)
        )

    retrieved_context = format_retrieved_context(
        final_hits, text_chars_per_chunk=text_chars_per_chunk
    )

    messages = build_rag_messages(
        patient_case=patient_case,
        retrieved_context=retrieved_context,
        num_sources=len(final_hits) or 1,
        prompt_config=prompt_config,
    )

    t_gen_start = time.perf_counter()
    try:
        llm_response = llm_client.generate(messages)
    except Exception as e:  
        logger.exception("LLM generation failed for case %s", case_id)
        return {
            "case_id": case_id,
            "patient_case": patient_case,
            "mode": "rag",
            "answer_json": None,
            "answer_raw_text": "",
            "retrieved_chunks": retrieved_chunks,
            "llm": {
                "provider": llm_client.provider_key,
                "provider_type": llm_client.provider_type,
                "model_name": llm_client.model_name,
                "usage": None,
            },
            "citation_validation": {
                "valid": False,
                "errors": [f"llm_call_failed: {e}"],
                "citation_count": 0,
                "invalid_citation_count": 0,
                "citation_coverage_estimate": None,
            },
            "timing": {
                "retrieval_time_sec": round(timings_ms.get("total_retrieval_ms", 0) / 1000.0, 4),
                "generation_time_sec": round(time.perf_counter() - t_gen_start, 4),
                "total_time_sec": round(time.perf_counter() - t0, 4),
            },
            "errors": [f"llm_call_failed: {e}"],
            "created_at": _utc_now(),
            "extra": extra_meta or {},
        }
    t_gen = time.perf_counter() - t_gen_start

    answer_text = llm_response.get("text", "") or ""
    answer_json, parse_errs = parse_llm_json(answer_text)
    if parse_errs:
        errors.extend(parse_errs)

    citation_check = validate_citations(answer_json, retrieved_chunks)

    return {
        "case_id": case_id,
        "patient_case": patient_case,
        "mode": "rag",
        "context_selection": engine.context_selection,
        "candidate_top_k": engine.candidate_top_k,
        "final_top_k": engine.final_top_k,
        "answer_json": answer_json,
        "answer_raw_text": answer_text,
        "retrieved_chunks": retrieved_chunks,
        "llm": {
            "provider": llm_client.provider_key,
            "provider_type": llm_client.provider_type,
            "model_name": llm_client.model_name,
            "usage": llm_response.get("usage"),
        },
        "citation_validation": citation_check,
        "timing": {
            "retrieval_time_sec": round(timings_ms.get("total_retrieval_ms", 0) / 1000.0, 4),
            "generation_time_sec": round(t_gen, 4),
            "total_time_sec": round(time.perf_counter() - t0, 4),
        },
        "errors": errors,
        "created_at": _utc_now(),
        "extra": extra_meta or {},
    }


def generate_no_rag_answer(
    *,
    case_id: str,
    patient_case: str,
    llm_client: LLMClient,
    extra_meta: Optional[Dict[str, Any]] = None,
    prompt_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    t0 = time.perf_counter()
    errors: List[str] = []

    messages = build_no_rag_messages(patient_case, prompt_config=prompt_config)
    t_gen_start = time.perf_counter()
    try:
        llm_response = llm_client.generate(messages)
    except Exception as e:  
        logger.exception("LLM generation failed for case %s (no_rag)", case_id)
        return {
            "case_id": case_id,
            "patient_case": patient_case,
            "mode": "no_rag",
            "answer_json": None,
            "answer_raw_text": "",
            "retrieved_chunks": [],
            "llm": {
                "provider": llm_client.provider_key,
                "provider_type": llm_client.provider_type,
                "model_name": llm_client.model_name,
                "usage": None,
            },
            "citation_validation": {
                "valid": False,
                "errors": [f"llm_call_failed: {e}"],
                "citation_count": 0,
                "invalid_citation_count": 0,
                "citation_coverage_estimate": None,
            },
            "timing": {
                "retrieval_time_sec": 0.0,
                "generation_time_sec": round(time.perf_counter() - t_gen_start, 4),
                "total_time_sec": round(time.perf_counter() - t0, 4),
            },
            "errors": [f"llm_call_failed: {e}"],
            "created_at": _utc_now(),
            "extra": extra_meta or {},
        }
    t_gen = time.perf_counter() - t_gen_start

    answer_text = llm_response.get("text", "") or ""
    answer_json, parse_errs = parse_llm_json(answer_text)
    if parse_errs:
        errors.extend(parse_errs)

    
    citation_check = validate_citations(answer_json, retrieved_chunks=[])
    if isinstance(answer_json, dict):
        for sect, idx, c in _iter_citations(answer_json):
            citation_check["errors"].append(
                f"no_rag_unexpected_citation: {sect}[{idx}] -> {c}"
            )
            citation_check["invalid_citation_count"] += 1
            citation_check["valid"] = False

    return {
        "case_id": case_id,
        "patient_case": patient_case,
        "mode": "no_rag",
        "answer_json": answer_json,
        "answer_raw_text": answer_text,
        "retrieved_chunks": [],
        "llm": {
            "provider": llm_client.provider_key,
            "provider_type": llm_client.provider_type,
            "model_name": llm_client.model_name,
            "usage": llm_response.get("usage"),
        },
        "citation_validation": citation_check,
        "timing": {
            "retrieval_time_sec": 0.0,
            "generation_time_sec": round(t_gen, 4),
            "total_time_sec": round(time.perf_counter() - t0, 4),
        },
        "errors": errors,
        "created_at": _utc_now(),
        "extra": extra_meta or {},
    }




def load_cases(
    path: str | Path,
    *,
    case_id_field: Optional[str] = None,
    patient_case_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Загружает clinical cases или retrieval queries.

    Поведение по полям:
        - Если ``case_id_field`` / ``patient_case_field`` заданы — используются они.
        - Иначе пробуем ``case_id`` → ``query_id`` для id и
          ``patient_case`` → ``query`` для текста.

    Возвращает унифицированный список dict с полями
    ``case_id``, ``patient_case``, ``raw`` (исходная запись).
    """
    items = read_jsonl_list(path)
    out: List[Dict[str, Any]] = []
    for it in items:
        if case_id_field:
            case_id = it.get(case_id_field)
        else:
            case_id = it.get("case_id") or it.get("query_id")
        if not case_id:
            case_id = f"case-{uuid.uuid4().hex[:8]}"

        if patient_case_field:
            patient_case = it.get(patient_case_field) or ""
        else:
            patient_case = it.get("patient_case") or it.get("query") or ""

        if not patient_case:
            logger.warning("Case %s has empty patient_case/query, skipping", case_id)
            continue
        out.append({"case_id": case_id, "patient_case": patient_case, "raw": it})
    return out
