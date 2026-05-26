from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import write_json

logger = logging.getLogger(__name__)


COMPARISON_COLUMNS: Tuple[str, ...] = (
    "model_key",
    "model_name",
    "embedding_dim",
    "number_of_queries",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "recall_at_5",
    "precision_at_5",
    "mrr",
    "avg_query_time_ms",
    "avg_retrieval_time_ms",
)


def write_comparison_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COMPARISON_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COMPARISON_COLUMNS})


def write_comparison_json(rows: List[Dict[str, Any]], path: Path) -> None:
    write_json(path, rows)


def select_best_model(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    def _score(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        
        recall = float(row.get("recall_at_5", 0.0) or 0.0)
        mrr_v = float(row.get("mrr", 0.0) or 0.0)
        prec = float(row.get("precision_at_5", 0.0) or 0.0)
        avg_t = float(row.get("avg_retrieval_time_ms", 0.0) or 0.0)
        return (recall, mrr_v, prec, -avg_t)

    best = max(rows, key=_score)
    reason = "highest recall_at_5"
    
    same_recall = [r for r in rows if float(r.get("recall_at_5", 0.0) or 0.0) == float(best.get("recall_at_5", 0.0) or 0.0)]
    if len(same_recall) > 1:
        reason = "highest recall_at_5 and mrr"
        same_mrr = [r for r in same_recall if float(r.get("mrr", 0.0) or 0.0) == float(best.get("mrr", 0.0) or 0.0)]
        if len(same_mrr) > 1:
            reason = "highest recall_at_5, mrr, precision_at_5"

    return {
        "best_model_key": best["model_key"],
        "reason": reason,
        "metrics": {
            "recall_at_5": float(best.get("recall_at_5", 0.0) or 0.0),
            "precision_at_5": float(best.get("precision_at_5", 0.0) or 0.0),
            "mrr": float(best.get("mrr", 0.0) or 0.0),
            "hit_at_1": float(best.get("hit_at_1", 0.0) or 0.0),
            "hit_at_5": float(best.get("hit_at_5", 0.0) or 0.0),
            "hit_at_10": float(best.get("hit_at_10", 0.0) or 0.0),
            "avg_query_time_ms": float(best.get("avg_query_time_ms", 0.0) or 0.0),
            "avg_retrieval_time_ms": float(best.get("avg_retrieval_time_ms", 0.0) or 0.0),
        },
        "model_name": best.get("model_name"),
        "embedding_dim": best.get("embedding_dim"),
    }


def write_best_model(best: Dict[str, Any], path: Path) -> None:
    
    write_json(path, best)
