from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .io_utils import ensure_dir, read_jsonl_list, setup_logging, write_jsonl

logger = logging.getLogger(__name__)


VALID_CASE_TYPES = {
    "diagnosis",
    "treatment",
    "diagnosis_and_next_steps",
    "multi_document",
    "symptoms",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}

META_PATTERNS = [
    r"\bнайди?\s+(раздел|пункт|протокол|параграф|главу|часть)\b",
    r"\bкакой\s+(раздел|протокол|пункт|параграф|документ)\s+нужен\b",
    r"\bкакой\s+(раздел|протокол|пункт|параграф|документ)\b",
    r"\bв\s+каком\s+(разделе|документе|протоколе)\b",
    r"\bпо\s+какому\s+(разделу|протоколу)\b",
    r"\bгде\s+(описан|написано|сказано|описаны)\b",
    r"\bкакой\s+документ\s+нужен\b",
    r"\bкак\s+должен\s+называться\s+раздел\b",
]
META_RE = re.compile("|".join(META_PATTERNS), re.IGNORECASE)


def _clean_meta_phrasings(text: str) -> str:
    cleaned = META_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;-—")
    return cleaned


def _map_case_type(query_type: Optional[str], expected_document_ids: Sequence[str]) -> str:
    if expected_document_ids and len(set(expected_document_ids)) > 1:
        return "multi_document"
    qt = (query_type or "").strip().lower()
    if qt == "diagnosis":
        return "diagnosis"
    if qt == "symptoms":
        return "diagnosis"
    if qt == "treatment":
        return "treatment"
    if qt in {"procedure", "monitoring", "rehabilitation"}:
        return "diagnosis_and_next_steps"
    return "diagnosis_and_next_steps"


def _normalize_difficulty(difficulty: Optional[str]) -> str:
    d = (difficulty or "").strip().lower()
    return d if d in VALID_DIFFICULTIES else "medium"


def _extract_expected_source_evidence(
    query: Dict[str, Any],
) -> List[Dict[str, Any]]:
    src = query.get("source_evidence") or []
    out: List[Dict[str, Any]] = []
    why = query.get("comment") or ""
    for ev in src:
        if not isinstance(ev, dict):
            continue
        out.append(
            {
                "chunk_id": ev.get("chunk_id"),
                "document_id": ev.get("document_id"),
                "document_title": ev.get("document_title"),
                "section_id": ev.get("section_id"),
                "section_title": ev.get("section_title"),
                "label": ev.get("label"),
                "page_start": ev.get("page_start"),
                "page_end": ev.get("page_end"),
                "evidence_text_preview": ev.get("evidence_text_preview"),
                "why_relevant": why,
            }
        )
    return out


def convert_retrieval_queries_to_clinical_cases(
    queries: List[Dict[str, Any]],
    *,
    drop_meta_phrasings: bool = True,
    case_id_prefix: str = "case",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, q in enumerate(queries, start=1):
        qid = q.get("query_id") or q.get("case_id") or f"{case_id_prefix}{i:03d}"
        original = (q.get("query") or q.get("patient_case") or "").strip()
        patient_case = _clean_meta_phrasings(original) if drop_meta_phrasings else original

        if len(patient_case.split()) < 6:
            kws = q.get("expected_section_keywords") or []
            if kws:
                patient_case = (
                    patient_case
                    + (" " if patient_case else "")
                    + "Ключевые клинические темы: "
                    + ", ".join(kws[:6]) + "."
                )

        if not patient_case:
            logger.warning("Query %s has empty patient_case after cleaning, skipping", qid)
            continue

        case = {
            "case_id": str(qid),
            "patient_case": patient_case,
            "case_type": _map_case_type(q.get("query_type"), q.get("expected_document_ids") or []),
            "difficulty": _normalize_difficulty(q.get("difficulty")),
            "expected_document_ids": list(q.get("expected_document_ids") or []),
            "expected_chunk_ids": list(q.get("expected_chunk_ids") or []),
            "expected_diagnoses": [],
            "expected_source_evidence": _extract_expected_source_evidence(q),
            "notes": f"case created from retrieval query {qid}",
            "review_status": "auto_generated",
            "requires_human_review": True,
            "source": "auto_converted_from_retrieval_query",
            "original_query": original,
            "original_query_type": q.get("query_type"),
        }
        out.append(case)
    return out


def validate_clinical_cases(cases: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen_ids: Dict[str, int] = {}
    for i, c in enumerate(cases):
        cid = c.get("case_id")
        if not cid:
            errors.append(f"row {i}: missing case_id")
            continue
        if cid in seen_ids:
            errors.append(f"duplicate case_id={cid!r} at rows {seen_ids[cid]} and {i}")
        seen_ids[cid] = i
        if not (c.get("patient_case") or "").strip():
            errors.append(f"case {cid}: empty patient_case")
        ct = c.get("case_type")
        if ct and ct not in VALID_CASE_TYPES:
            errors.append(f"case {cid}: unknown case_type={ct!r}")
        diff = c.get("difficulty")
        if diff and diff not in VALID_DIFFICULTIES:
            errors.append(f"case {cid}: unknown difficulty={diff!r}")
    return errors


def merge_with_manual_cases(
    auto_cases: List[Dict[str, Any]],
    manual_cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {c["case_id"]: c for c in auto_cases if c.get("case_id")}
    for m in manual_cases:
        cid = m.get("case_id")
        if not cid:
            logger.warning("Manual case without case_id, skipping: %s", m)
            continue
        if cid in by_id:
            logger.info("Manual case %s overrides auto-generated case", cid)
        m.setdefault("review_status", "manual")
        m.setdefault("requires_human_review", False)
        m.setdefault("source", "manual")
        by_id[cid] = m
    return list(by_id.values())


def write_review_template_csv(cases: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    cols = [
        "case_id",
        "case_type",
        "difficulty",
        "review_status",
        "requires_human_review",
        "expected_diagnoses",
        "patient_case",
        "original_query",
        "notes",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for c in cases:
            writer.writerow(
                {
                    "case_id": c.get("case_id", ""),
                    "case_type": c.get("case_type", ""),
                    "difficulty": c.get("difficulty", ""),
                    "review_status": c.get("review_status", ""),
                    "requires_human_review": c.get("requires_human_review", ""),
                    "expected_diagnoses": "|".join(c.get("expected_diagnoses") or []),
                    "patient_case": (c.get("patient_case") or "")[:1000],
                    "original_query": (c.get("original_query") or "")[:1000],
                    "notes": c.get("notes", ""),
                }
            )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert retrieval eval queries → clinical cases (+ optional manual cases)."
    )
    p.add_argument(
        "--input",
        required=True,
        type=str,
        help="JSONL с retrieval queries (например, data/retrieval_eval_queries_plus_hard_v1.jsonl).",
    )
    p.add_argument(
        "--manual",
        default=None,
        type=str,
        help="(Опционально) JSONL с ручными hard-кейсами; склеиваются с авто.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=str,
        help="Куда писать итоговый clinical_cases_v1.jsonl.",
    )
    p.add_argument(
        "--review-csv",
        default=None,
        type=str,
        help="(Опционально) куда писать CSV-шаблон для ручного ревью.",
    )
    p.add_argument(
        "--mode",
        choices=["convert"],
        default="convert",
        help="Сейчас поддерживается только 'convert'.",
    )
    p.add_argument(
        "--keep-meta-phrasings",
        action="store_true",
        help="НЕ чистить мета-формулировки 'найди раздел / какой протокол'.",
    )
    p.add_argument("--limit", type=int, default=None, help="Ограничить число кейсов.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    setup_logging(verbose=args.verbose, log_file=None, name="clinical_cases_tools")

    queries = read_jsonl_list(args.input)
    logger.info("Loaded %d retrieval queries from %s", len(queries), args.input)

    auto = convert_retrieval_queries_to_clinical_cases(
        queries, drop_meta_phrasings=not args.keep_meta_phrasings
    )
    logger.info("Auto-generated %d clinical cases", len(auto))

    cases: List[Dict[str, Any]] = auto
    if args.manual:
        manual = read_jsonl_list(args.manual)
        logger.info("Loaded %d manual cases from %s", len(manual), args.manual)
        cases = merge_with_manual_cases(auto, manual)
        logger.info("After merge: %d total cases", len(cases))

    if args.limit:
        cases = cases[: args.limit]
        logger.info("Limited to %d cases", len(cases))

    errors = validate_clinical_cases(cases)
    if errors:
        for e in errors:
            logger.error("VALIDATION: %s", e)
        return 2

    n = write_jsonl(out_path, cases)
    logger.info("Wrote %d cases → %s", n, out_path)

    if args.review_csv:
        write_review_template_csv(cases, args.review_csv)
        logger.info("Wrote review template → %s", args.review_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
