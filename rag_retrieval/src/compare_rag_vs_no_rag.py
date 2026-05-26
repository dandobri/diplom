from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import ensure_dir, read_jsonl_list, setup_logging, write_jsonl

logger = logging.getLogger(__name__)


SUMMARY_COLUMNS = [
    "case_id",
    "patient_case",
    "rag_answer_valid_json",
    "no_rag_answer_valid_json",
    "rag_citation_count",
    "rag_invalid_citation_count",
    "rag_citation_coverage_estimate",
    "no_rag_citation_count",
    "rag_num_diagnoses",
    "no_rag_num_diagnoses",
    "rag_insufficient_information",
    "no_rag_insufficient_information",
    "rag_retrieved_document_ids",
    "rag_retrieved_chunk_ids",
    "rag_model",
    "no_rag_model",
]


def _index_by_case_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in records:
        cid = r.get("case_id")
        if cid is None:
            continue
        out[str(cid)] = r
    return out


def _num_diagnoses(rec: Dict[str, Any]) -> int:
    aj = rec.get("answer_json") if isinstance(rec, dict) else None
    if not isinstance(aj, dict):
        return 0
    diag = aj.get("differential_diagnoses")
    return len(diag) if isinstance(diag, list) else 0


def _insufficient(rec: Dict[str, Any]) -> Optional[bool]:
    aj = rec.get("answer_json") if isinstance(rec, dict) else None
    if not isinstance(aj, dict):
        return None
    val = aj.get("insufficient_information")
    return bool(val) if isinstance(val, bool) else None


def _retrieved_doc_ids(rec: Dict[str, Any]) -> List[str]:
    chunks = rec.get("retrieved_chunks") or []
    seen: List[str] = []
    for c in chunks:
        d = c.get("document_id")
        if d and d not in seen:
            seen.append(str(d))
    return seen


def _retrieved_chunk_ids(rec: Dict[str, Any]) -> List[str]:
    return [c.get("chunk_id") for c in (rec.get("retrieved_chunks") or []) if c.get("chunk_id")]


def build_pair(rag: Optional[Dict[str, Any]], no_rag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pivot = rag or no_rag or {}
    return {
        "case_id": pivot.get("case_id"),
        "patient_case": pivot.get("patient_case"),
        "rag": rag,
        "no_rag": no_rag,
    }


def build_summary_row(pair: Dict[str, Any]) -> Dict[str, Any]:
    rag = pair.get("rag") or {}
    no_rag = pair.get("no_rag") or {}
    rag_cv = (rag.get("citation_validation") or {}) if isinstance(rag, dict) else {}
    no_rag_cv = (no_rag.get("citation_validation") or {}) if isinstance(no_rag, dict) else {}

    return {
        "case_id": pair.get("case_id"),
        "patient_case": (pair.get("patient_case") or "")[:500],
        "rag_answer_valid_json": isinstance(rag.get("answer_json"), dict),
        "no_rag_answer_valid_json": isinstance(no_rag.get("answer_json"), dict),
        "rag_citation_count": rag_cv.get("citation_count", 0),
        "rag_invalid_citation_count": rag_cv.get("invalid_citation_count", 0),
        "rag_citation_coverage_estimate": rag_cv.get("citation_coverage_estimate"),
        "no_rag_citation_count": no_rag_cv.get("citation_count", 0),
        "rag_num_diagnoses": _num_diagnoses(rag),
        "no_rag_num_diagnoses": _num_diagnoses(no_rag),
        "rag_insufficient_information": _insufficient(rag),
        "no_rag_insufficient_information": _insufficient(no_rag),
        "rag_retrieved_document_ids": "|".join(_retrieved_doc_ids(rag)),
        "rag_retrieved_chunk_ids": "|".join(_retrieved_chunk_ids(rag)),
        "rag_model": (rag.get("llm") or {}).get("model_name"),
        "no_rag_model": (no_rag.get("llm") or {}).get("model_name"),
    }


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in SUMMARY_COLUMNS})


def merge(
    rag_path: str | Path,
    no_rag_path: str | Path,
    output_pairs: str | Path,
    output_summary: str | Path,
) -> Dict[str, Any]:
    rag_records = read_jsonl_list(rag_path)
    no_rag_records = read_jsonl_list(no_rag_path)
    logger.info("Loaded RAG=%d no_RAG=%d", len(rag_records), len(no_rag_records))

    rag_idx = _index_by_case_id(rag_records)
    no_rag_idx = _index_by_case_id(no_rag_records)

    all_ids = list(rag_idx.keys())
    for cid in no_rag_idx:
        if cid not in rag_idx:
            all_ids.append(cid)

    pairs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for cid in all_ids:
        pair = build_pair(rag_idx.get(cid), no_rag_idx.get(cid))
        pairs.append(pair)
        rows.append(build_summary_row(pair))

    write_jsonl(output_pairs, pairs)
    write_summary_csv(rows, Path(output_summary))
    logger.info(
        "Wrote pairs=%s summary=%s (cases=%d)",
        output_pairs,
        output_summary,
        len(pairs),
    )
    return {
        "num_cases": len(pairs),
        "rag_only": sum(1 for p in pairs if p["rag"] and not p["no_rag"]),
        "no_rag_only": sum(1 for p in pairs if p["no_rag"] and not p["rag"]),
        "both": sum(1 for p in pairs if p["rag"] and p["no_rag"]),
        "output_pairs": str(output_pairs),
        "output_summary": str(output_summary),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge RAG and no-RAG outputs into pairs + summary CSV")
    p.add_argument("--rag-path", required=True, type=str)
    p.add_argument("--no-rag-path", required=True, type=str)
    p.add_argument("--output-pairs", default="outputs_generation/rag_vs_no_rag_pairs.jsonl")
    p.add_argument("--output-summary", default="outputs_generation/rag_vs_no_rag_summary.csv")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    ensure_dir(Path(args.output_pairs).parent)
    setup_logging(
        verbose=args.verbose,
        log_file=Path(args.output_pairs).parent / "compare_rag_vs_no_rag.log",
        name="compare",
    )
    info = merge(args.rag_path, args.no_rag_path, args.output_pairs, args.output_summary)
    logger.info("Merge stats: %s", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
