from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .retrieval import RetrievalHit

logger = logging.getLogger(__name__)


def _to_set(values: Optional[Iterable[Any]]) -> set:
    if not values:
        return set()
    return {v for v in values if v is not None}


def hit_at_k(hits: Sequence[RetrievalHit], expected_doc_ids: Iterable[str], k: int) -> bool:
    
    expected = _to_set(expected_doc_ids)
    if not expected:
        return False
    for h in hits[:k]:
        if h.document_id in expected:
            return True
    return False


def recall_at_k(
    hits: Sequence[RetrievalHit], expected_doc_ids: Iterable[str], k: int
) -> float:
    
    expected = _to_set(expected_doc_ids)
    if not expected:
        return 0.0
    found = {h.document_id for h in hits[:k] if h.document_id in expected}
    return len(found) / len(expected)


def precision_at_k(
    hits: Sequence[RetrievalHit], expected_doc_ids: Iterable[str], k: int
) -> float:
    """Доля результатов в top-k, попавших в expected.

    Знаменатель — фиксированный k (как требует ТЗ). Это корректно, когда
    в индексе ≥ k чанков; на нашем корпусе 2000–5000 это всегда так.
    """
    expected = _to_set(expected_doc_ids)
    if not expected or k <= 0:
        return 0.0
    relevant = sum(1 for h in hits[:k] if h.document_id in expected)
    return relevant / k


def first_relevant_rank(
    hits: Sequence[RetrievalHit], expected_doc_ids: Iterable[str]
) -> Optional[int]:
    expected = _to_set(expected_doc_ids)
    if not expected:
        return None
    for h in hits:
        if h.document_id in expected:
            return h.rank
    return None


def mrr(hits: Sequence[RetrievalHit], expected_doc_ids: Iterable[str]) -> float:
    
    rank = first_relevant_rank(hits, expected_doc_ids)
    if rank is None:
        return 0.0
    return 1.0 / rank




def _normalize_text(s: Optional[str]) -> str:
    return (s or "").lower()


def section_keyword_hit_at_k(
    hits: Sequence[RetrievalHit],
    expected_keywords: Optional[Iterable[str]],
    k: int,
) -> Optional[bool]:
    """True, если хотя бы в одном из top-k встречается хотя бы одно ключевое слово.

    Возвращает None, если expected_keywords пуст (метрика не применима).
    Поиск идет по section_title и text (case-insensitive).
    """
    if not expected_keywords:
        return None
    kws = [kw.lower() for kw in expected_keywords if kw]
    if not kws:
        return None
    for h in hits[:k]:
        haystack = _normalize_text(h.section_title) + "\n" + _normalize_text(h.text)
        if any(kw in haystack for kw in kws):
            return True
    return False




def evaluate_query(
    hits: Sequence[RetrievalHit],
    expected_document_ids: Iterable[str],
    expected_section_keywords: Optional[Iterable[str]] = None,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    """Считает все метрики для одного запроса.

    Returns:
        Словарь с числовыми метриками и first_relevant_rank.
    """
    expected = list(expected_document_ids or [])
    out: Dict[str, Any] = {
        "expected_document_ids": expected,
    }
    for k in ks:
        out[f"hit_at_{k}"] = hit_at_k(hits, expected, k)
    out["recall_at_5"] = recall_at_k(hits, expected, 5)
    out["precision_at_5"] = precision_at_k(hits, expected, 5)
    out["mrr"] = mrr(hits, expected)
    out["first_relevant_rank"] = first_relevant_rank(hits, expected)

    
    sec_hit = section_keyword_hit_at_k(hits, expected_section_keywords, k=5)
    if sec_hit is not None:
        out["section_keyword_hit_at_5"] = sec_hit
    return out




def aggregate_metrics(per_query: List[Dict[str, Any]]) -> Dict[str, float]:
    
    n = len(per_query)
    if n == 0:
        return {}

    def _avg(field: str) -> float:
        values = [float(q[field]) for q in per_query if field in q]
        return sum(values) / len(values) if values else 0.0

    agg = {
        "number_of_queries": n,
        "hit_at_1": _avg("hit_at_1"),
        "hit_at_5": _avg("hit_at_5"),
        "hit_at_10": _avg("hit_at_10"),
        "recall_at_5": _avg("recall_at_5"),
        "precision_at_5": _avg("precision_at_5"),
        "mrr": _avg("mrr"),
    }
    
    sec_values = [q["section_keyword_hit_at_5"] for q in per_query if "section_keyword_hit_at_5" in q]
    if sec_values:
        agg["section_keyword_hit_at_5"] = sum(float(v) for v in sec_values) / len(sec_values)
        agg["section_keyword_coverage"] = len(sec_values) / n
    return agg
