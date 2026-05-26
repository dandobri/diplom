from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .retrieval import RetrievalHit

logger = logging.getLogger(__name__)


def _to_set(values: Optional[Iterable[Any]]) -> Set[Any]:
    if not values:
        return set()
    return {v for v in values if v not in (None, "")}


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


def is_document_relevant(hit: RetrievalHit, query_item: Dict[str, Any]) -> bool:
    """Документ-уровень. Если разметки нет — возвращает False (а не None),
    так как expected_document_ids — обязательное поле для оценки retrieval."""
    expected = _to_set(query_item.get("expected_document_ids"))
    if not expected:
        return False
    return hit.document_id in expected


def is_chunk_relevant(
    hit: RetrievalHit, query_item: Dict[str, Any]
) -> Optional[bool]:
    expected = _to_set(query_item.get("expected_chunk_ids"))
    if not expected:
        return None
    return hit.chunk_id in expected


def is_section_relevant(
    hit: RetrievalHit, query_item: Dict[str, Any]
) -> Optional[bool]:
    expected_section_ids = _to_set(query_item.get("expected_section_ids"))
    expected_section_titles = _to_set(query_item.get("expected_section_titles"))
    expected_keywords = [
        kw for kw in (query_item.get("expected_section_keywords") or []) if kw
    ]
    if not (expected_section_ids or expected_section_titles or expected_keywords):
        return None

    expected_docs = _to_set(query_item.get("expected_document_ids"))
    if expected_docs and hit.document_id not in expected_docs:
        return False

    if expected_section_ids and hit.section_id is not None:
        if str(hit.section_id) in {str(x) for x in expected_section_ids}:
            return True

    if expected_section_titles and hit.section_title:
        if hit.section_title in expected_section_titles:
            return True

    if expected_keywords:
        haystack = (hit.section_title or "").lower() + "\n" + (hit.text or "").lower()
        for kw in expected_keywords:
            if kw and kw.lower() in haystack:
                return True

    return False


def is_page_relevant(
    hit: RetrievalHit,
    query_item: Dict[str, Any],
    *,
    soft: bool = False,
    page_tolerance: int = 1,
) -> Optional[bool]:
    evidences = query_item.get("source_evidence") or []
    if not evidences:
        return None

    has_pages = False
    tolerance = page_tolerance if soft else 0
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        ev_doc = ev.get("document_id")
        ev_ps = ev.get("page_start")
        ev_pe = ev.get("page_end")
        if ev_ps is None and ev_pe is None:
            continue
        has_pages = True
        if ev_doc and hit.document_id and ev_doc != hit.document_id:
            continue
        if _ranges_overlap(
            hit.page_start,
            hit.page_end,
            ev_ps,
            ev_pe,
            tolerance=tolerance,
        ):
            return True

    if not has_pages:
        return None
    return False


def is_label_relevant(
    hit: RetrievalHit, query_item: Dict[str, Any]
) -> Optional[bool]:
    expected = _to_set(query_item.get("expected_labels"))
    if not expected:
        return None
    if hit.label is None:
        return False
    return hit.label in expected

def hit_at_k(
    hits: Sequence[RetrievalHit],
    query_item: Dict[str, Any],
    relevance_fn,
    k: int,
) -> Optional[bool]:
    """True если в top-k есть релевантный; None если разметки нет ни для одного."""
    any_hit = False
    any_marked = False
    for h in hits[:k]:
        rel = relevance_fn(h, query_item)
        if rel is None:
            continue
        any_marked = True
        if rel is True:
            any_hit = True
            break 
    if not any_marked:
        return None
    return any_hit


def precision_at_k(
    hits: Sequence[RetrievalHit],
    query_item: Dict[str, Any],
    relevance_fn,
    k: int,
) -> Optional[float]:
    if k <= 0:
        return None
    relevant = 0
    none_count = 0
    for h in hits[:k]:
        rel = relevance_fn(h, query_item)
        if rel is None:
            none_count += 1
        elif rel is True:
            relevant += 1
    if none_count >= len(hits[:k]):
        return None
    return relevant / k


def mrr(
    hits: Sequence[RetrievalHit],
    query_item: Dict[str, Any],
    relevance_fn,
) -> Optional[float]:
    any_marked = False
    for h in hits:
        rel = relevance_fn(h, query_item)
        if rel is None:
            continue
        any_marked = True
        if rel is True:
            return 1.0 / h.rank
    return None if not any_marked else 0.0


def document_recall_at_k(
    hits: Sequence[RetrievalHit], query_item: Dict[str, Any], k: int
) -> Optional[float]:
    expected = _to_set(query_item.get("expected_document_ids"))
    if not expected:
        return None
    found = {h.document_id for h in hits[:k] if h.document_id in expected}
    return len(found) / len(expected)


def chunk_recall_at_k(
    hits: Sequence[RetrievalHit], query_item: Dict[str, Any], k: int
) -> Optional[float]:
    expected = _to_set(query_item.get("expected_chunk_ids"))
    if not expected:
        return None
    found = {h.chunk_id for h in hits[:k] if h.chunk_id in expected}
    return len(found) / len(expected)

def mean_skip_none(values: Sequence[Optional[float]]) -> Tuple[Optional[float], int]:
    """Среднее по non-None значениям. Возвращает (mean, coverage)."""
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None, 0
    return sum(valid) / len(valid), len(valid)

def evaluate_query_advanced(
    hits: Sequence[RetrievalHit],
    query_item: Dict[str, Any],
    *,
    final_top_k: int = 5,
    page_tolerance: int = 1,
) -> Dict[str, Any]:
    k = final_top_k

    out: Dict[str, Any] = {}

    out["document_hit_at_1"] = hit_at_k(hits, query_item, is_document_relevant, 1)
    out["document_hit_at_5"] = hit_at_k(hits, query_item, is_document_relevant, k)
    if len(hits) >= 10:
        out["document_hit_at_10"] = hit_at_k(hits, query_item, is_document_relevant, 10)
    out["document_recall_at_5"] = document_recall_at_k(hits, query_item, k)
    out["document_precision_at_5"] = precision_at_k(hits, query_item, is_document_relevant, k)
    out["document_mrr"] = mrr(hits, query_item, is_document_relevant)

    out["chunk_hit_at_1"] = hit_at_k(hits, query_item, is_chunk_relevant, 1)
    out["chunk_hit_at_5"] = hit_at_k(hits, query_item, is_chunk_relevant, k)
    out["chunk_recall_at_5"] = chunk_recall_at_k(hits, query_item, k)
    out["chunk_precision_at_5"] = precision_at_k(hits, query_item, is_chunk_relevant, k)
    out["chunk_mrr"] = mrr(hits, query_item, is_chunk_relevant)

    out["section_hit_at_5"] = hit_at_k(hits, query_item, is_section_relevant, k)
    out["section_precision_at_5"] = precision_at_k(hits, query_item, is_section_relevant, k)
    out["section_mrr"] = mrr(hits, query_item, is_section_relevant)

    page_fn = is_page_relevant
    soft_fn = lambda h, q: is_page_relevant(h, q, soft=True, page_tolerance=page_tolerance)
    out["page_hit_at_1"] = hit_at_k(hits, query_item, page_fn, 1)
    out["page_hit_at_5"] = hit_at_k(hits, query_item, page_fn, k)
    out["page_precision_at_5"] = precision_at_k(hits, query_item, page_fn, k)
    out["page_mrr"] = mrr(hits, query_item, page_fn)
    out["soft_page_hit_at_5"] = hit_at_k(hits, query_item, soft_fn, k)

    out["label_hit_at_5"] = hit_at_k(hits, query_item, is_label_relevant, k)

    return out


def aggregate_advanced(
    per_query: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Усредняет метрики по корпусу запросов, пропуская None.

    Возвращает словарь {metric_name: float или None} плюс {metric_name + '_coverage': int}.
    """
    if not per_query:
        return {}
    keys = set()
    for q in per_query:
        keys.update(q.keys())
    agg: Dict[str, Any] = {}
    for key in sorted(keys):
        values = [q.get(key) for q in per_query]
        cleaned: List[Optional[float]] = []
        for v in values:
            if v is None:
                cleaned.append(None)
            elif isinstance(v, bool):
                cleaned.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                cleaned.append(float(v))
            else:
                cleaned.append(None)
        mean, coverage = mean_skip_none(cleaned)
        agg[key] = round(mean, 4) if mean is not None else None
        agg[f"{key}_coverage"] = coverage
    agg["number_of_queries"] = len(per_query)
    return agg
