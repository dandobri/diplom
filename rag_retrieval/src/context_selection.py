from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .retrieval import RetrievalHit

logger = logging.getLogger(__name__)


SUPPORTED_MODES = ("none", "anchor_document", "anchor_section", "anchor_page")

NEIGHBOR_DELTAS = (1, -1, 2, -2, 3, -3)


def _ranges_overlap(
    a_start: Optional[int],
    a_end: Optional[int],
    b_start: Optional[int],
    b_end: Optional[int],
    *,
    tolerance: int = 0,
) -> bool:
    if a_start is None or b_start is None:
        return False
    a_e = a_end if a_end is not None else a_start
    b_e = b_end if b_end is not None else b_start
    return a_start <= (b_e + tolerance) and (a_e + tolerance) >= b_start


def _meta_to_hit(meta: Dict[str, Any], context_source: str) -> RetrievalHit:
    extra = {
        "context_source": context_source,
        "chunk_index": meta.get("chunk_index"),
        "specialty": meta.get("specialty"),
        "stage": meta.get("stage"),
        "content_hash": meta.get("content_hash"),
    }
    return RetrievalHit(
        rank=0,
        score=0.0,
        chunk_id=str(meta.get("id") or ""),
        document_id=meta.get("document_id"),
        document_title=meta.get("document_title"),
        section_id=meta.get("section_id"),
        section_title=meta.get("section_title"),
        label=meta.get("label"),
        source=meta.get("source"),
        page_start=meta.get("page_start"),
        page_end=meta.get("page_end"),
        text=meta.get("text") or "",
        extra=extra,
    )


def _copy_hit(hit: RetrievalHit, context_source: str) -> RetrievalHit:
    new = RetrievalHit(
        rank=hit.rank,
        score=hit.score,
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        document_title=hit.document_title,
        section_id=hit.section_id,
        section_title=hit.section_title,
        label=hit.label,
        source=hit.source,
        page_start=hit.page_start,
        page_end=hit.page_end,
        text=hit.text,
        extra=dict(hit.extra) if isinstance(hit.extra, dict) else {},
    )
    new.extra["context_source"] = context_source
    return new


def select_context_top_k(
    *,
    reranked: Sequence[RetrievalHit],
    reranker_scores: Sequence[float],
    dense_ranks_in_rerank: Sequence[int],
    metadata: Sequence[Dict[str, Any]],
    mode: str,
    final_top_k: int,
    page_tolerance: int = 1,
) -> Tuple[List[RetrievalHit], List[Optional[float]], List[Optional[int]]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unknown context-selection mode: {mode!r}. "
            f"Supported: {SUPPORTED_MODES}"
        )

    if not reranked or final_top_k <= 0:
        return [], [], []

    score_map: Dict[str, Tuple[float, int]] = {
        h.chunk_id: (float(s), int(d))
        for h, s, d in zip(reranked, reranker_scores, dense_ranks_in_rerank)
        if h.chunk_id
    }

    if mode == "none":
        return _take_topk_reranker(
            reranked, reranker_scores, dense_ranks_in_rerank, final_top_k
        )

    anchor = reranked[0]
    selected_ids: set = set()
    selected: List[RetrievalHit] = []
    selected_scores: List[Optional[float]] = []
    selected_dranks: List[Optional[int]] = []

    def _add(hit: RetrievalHit, source_label: str) -> bool:
        if len(selected) >= final_top_k:
            return False
        if not hit.chunk_id or hit.chunk_id in selected_ids:
            return False
        new_hit = _copy_hit(hit, source_label)
        new_hit.rank = len(selected) + 1
        sc, dr = score_map.get(hit.chunk_id, (None, None))
        if sc is not None:
            new_hit.extra["reranker_score"] = sc
        if dr is not None:
            new_hit.extra["dense_rank"] = dr
        selected.append(new_hit)
        selected_scores.append(sc)
        selected_dranks.append(dr)
        selected_ids.add(hit.chunk_id)
        return True

    def _matches_scope(
        doc_id: Optional[str],
        section_id: Optional[Any],
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> bool:
        if doc_id != anchor.document_id:
            return False
        if mode == "anchor_section":
            return str(section_id or "") == str(anchor.section_id or "")
        if mode == "anchor_page":
            return _ranges_overlap(
                page_start, page_end,
                anchor.page_start, anchor.page_end,
                tolerance=page_tolerance,
            )
        return True

    _add(anchor, "anchor")

    for h in reranked[1:]:
        if len(selected) >= final_top_k:
            break
        if _matches_scope(h.document_id, h.section_id, h.page_start, h.page_end):
            _add(h, "reranker")

    anchor_idx = anchor.extra.get("chunk_index") if isinstance(anchor.extra, dict) else None
    if anchor_idx is not None and len(selected) < final_top_k:
        for delta in NEIGHBOR_DELTAS:
            if len(selected) >= final_top_k:
                break
            target_idx = int(anchor_idx) + int(delta)
            for m in metadata:
                if m.get("document_id") != anchor.document_id:
                    continue
                if m.get("chunk_index") != target_idx:
                    continue
                if not _matches_scope(
                    m.get("document_id"),
                    m.get("section_id"),
                    m.get("page_start"),
                    m.get("page_end"),
                ):
                    continue
                _add(_meta_to_hit(m, "neighbor"), "neighbor")

    if (
        mode == "anchor_document"
        and anchor.page_start is not None
        and len(selected) < final_top_k
    ):
        for m in metadata:
            if len(selected) >= final_top_k:
                break
            if m.get("document_id") != anchor.document_id:
                continue
            if not _ranges_overlap(
                m.get("page_start"), m.get("page_end"),
                anchor.page_start, anchor.page_end,
                tolerance=page_tolerance,
            ):
                continue
            _add(_meta_to_hit(m, "same_page"), "same_page")

    if (
        mode in ("anchor_document", "anchor_page")
        and anchor.section_id not in (None, "")
        and len(selected) < final_top_k
    ):
        for m in metadata:
            if len(selected) >= final_top_k:
                break
            if m.get("document_id") != anchor.document_id:
                continue
            if str(m.get("section_id") or "") != str(anchor.section_id or ""):
                continue
            _add(_meta_to_hit(m, "same_section"), "same_section")

    if mode == "anchor_document" and len(selected) < final_top_k:
        same_doc_metas = [
            m for m in metadata if m.get("document_id") == anchor.document_id
        ]
        if anchor_idx is not None:
            same_doc_metas.sort(
                key=lambda m: abs(int(m.get("chunk_index") or 0) - int(anchor_idx))
            )
        for m in same_doc_metas:
            if len(selected) >= final_top_k:
                break
            _add(_meta_to_hit(m, "same_document"), "same_document")

    if len(selected) < final_top_k:
        for h in reranked[1:]:
            if len(selected) >= final_top_k:
                break
            _add(h, "reranker_fallback")

    return selected, selected_scores, selected_dranks


def _take_topk_reranker(
    reranked: Sequence[RetrievalHit],
    reranker_scores: Sequence[float],
    dense_ranks_in_rerank: Sequence[int],
    final_top_k: int,
) -> Tuple[List[RetrievalHit], List[Optional[float]], List[Optional[int]]]:
    out_h: List[RetrievalHit] = []
    out_s: List[Optional[float]] = []
    out_d: List[Optional[int]] = []
    for new_rank, (h, s, d) in enumerate(
        zip(reranked[:final_top_k], reranker_scores[:final_top_k],
            dense_ranks_in_rerank[:final_top_k]),
        start=1,
    ):
        new_h = _copy_hit(h, "reranker")
        new_h.rank = new_rank
        new_h.extra["reranker_score"] = float(s)
        new_h.extra["dense_rank"] = int(d)
        out_h.append(new_h)
        out_s.append(float(s))
        out_d.append(int(d))
    return out_h, out_s, out_d
