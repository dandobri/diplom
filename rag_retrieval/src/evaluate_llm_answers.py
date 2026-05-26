from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from .io_utils import (
    ensure_dir,
    read_jsonl_list,
    setup_logging,
    write_json,
    write_jsonl,
)
from .llm_client import LLMClient
from .llm_eval_metrics import (
    aggregate_global,
    aggregate_per_case,
    compute_citation_metrics_for_rag,
    match_diagnoses,
)
from .llm_judge import (
    JudgeRunner,
    extract_claims,
    format_reference_context,
    run_answer_relevance_judge,
    run_citation_accuracy_judge,
    run_faithfulness_judge,
)
from .prompt_templates import load_prompt_config

logger = logging.getLogger(__name__)

def _open_append(path: Path, *, overwrite: bool, resume: bool) -> Any:
    if path.exists() and not (overwrite or resume):
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite to replace or --resume to append."
        )
    if overwrite and path.exists():
        path.unlink()
    return path.open("a", encoding="utf-8")


def _write_record(fh: Any, record: Dict[str, Any]) -> None:
    fh.write(json.dumps(record, ensure_ascii=False))
    fh.write("\n")
    fh.flush()


def _existing_case_mode_keys(path: Path) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()
    seen: Set[Tuple[str, str]] = set()
    try:
        for rec in read_jsonl_list(path):
            cid = rec.get("case_id")
            mode = rec.get("mode")
            if cid and mode:
                seen.add((str(cid), str(mode)))
    except Exception as e:
        logger.warning("Could not parse existing %s: %s — treating as empty.", path, e)
    return seen


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_pairs(
    *,
    pairs_path: Optional[Path],
    rag_path: Optional[Path],
    no_rag_path: Optional[Path],
) -> List[Dict[str, Any]]:
    if pairs_path:
        recs = read_jsonl_list(pairs_path)
        out: List[Dict[str, Any]] = []
        for r in recs:
            if not isinstance(r, dict):
                continue
            case_id = r.get("case_id")
            if not case_id:
                pivot = r.get("rag") or r.get("no_rag") or {}
                case_id = pivot.get("case_id")
            patient_case = r.get("patient_case")
            if not patient_case:
                pivot = r.get("rag") or r.get("no_rag") or {}
                patient_case = pivot.get("patient_case")
            out.append({
                "case_id": str(case_id) if case_id else None,
                "patient_case": patient_case or "",
                "rag": r.get("rag"),
                "no_rag": r.get("no_rag"),
            })
        return out

    if not (rag_path or no_rag_path):
        raise ValueError("Either --pairs-path or --rag-answers-path/--no-rag-answers-path is required")

    rag_recs = read_jsonl_list(rag_path) if rag_path else []
    no_rag_recs = read_jsonl_list(no_rag_path) if no_rag_path else []

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rag_recs:
        cid = str(r.get("case_id") or "")
        if cid:
            by_id.setdefault(cid, {"case_id": cid, "patient_case": r.get("patient_case") or "", "rag": None, "no_rag": None})
            by_id[cid]["rag"] = r
    for r in no_rag_recs:
        cid = str(r.get("case_id") or "")
        if cid:
            by_id.setdefault(cid, {"case_id": cid, "patient_case": r.get("patient_case") or "", "rag": None, "no_rag": None})
            by_id[cid]["no_rag"] = r
            if not by_id[cid]["patient_case"]:
                by_id[cid]["patient_case"] = r.get("patient_case") or ""

    return list(by_id.values())


def _load_clinical_cases(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in read_jsonl_list(path):
        cid = c.get("case_id")
        if cid:
            by_id[str(cid)] = c
    return by_id

def _evaluate_one_mode(
    *,
    case_id: str,
    mode: str,
    patient_case: str,
    answer_record: Dict[str, Any],
    reference_chunks: Sequence[Dict[str, Any]],
    reference_context: str,
    faithfulness_runner: JudgeRunner,
    citation_runner: Optional[JudgeRunner],
    relevance_runner: Optional[JudgeRunner],
    expected_diagnoses: Sequence[str],
    max_claims_per_answer: int,
    fh_claims: Any,
    fh_failed: Any,
) -> Dict[str, Any]:
    answer_json = (answer_record or {}).get("answer_json")
    valid_json = isinstance(answer_json, dict)

    claims = extract_claims(
        answer_json if valid_json else None,
        case_id=case_id,
        mode=mode,
        max_claims=max_claims_per_answer,
    )

    judge_errors: List[str] = []
    timings: Dict[str, float] = {}

    f_eval = run_faithfulness_judge(
        faithfulness_runner,
        case_id=case_id,
        mode=mode,
        patient_case=patient_case,
        claims=claims,
        reference_context=reference_context,
    )
    timings["faithfulness_judge_sec"] = f_eval.get("elapsed_sec") or 0.0
    if f_eval.get("errors"):
        judge_errors.extend([f"faithfulness:{e}" for e in f_eval["errors"]])

    f_parsed = f_eval.get("parsed") or {}
    claim_evals_raw = f_parsed.get("claim_evaluations") if isinstance(f_parsed, dict) else None
    claim_evals: List[Dict[str, Any]] = []

    if isinstance(claim_evals_raw, list):
        by_claim_id: Dict[str, Dict[str, Any]] = {}
        for ce in claim_evals_raw:
            if isinstance(ce, dict):
                cid = ce.get("claim_id")
                if cid:
                    by_claim_id[str(cid)] = ce

        for c in claims:
            cid = c["claim_id"]
            ce = by_claim_id.get(cid) or {}
            claim_evals.append({
                "case_id": case_id,
                "mode": mode,
                "claim_id": cid,
                "claim_type": c["claim_type"],
                "claim_text": c["claim_text"],
                "citations": c["citations"],
                "source_item_path": c["source_item_path"],
                "support_status": ce.get("support_status"),
                "support_score": ce.get("support_score"),
                "citation_supports_claim": ce.get("citation_supports_claim"),
                "uses_valid_citation": ce.get("uses_valid_citation"),
                "supporting_source_ids": ce.get("supporting_source_ids") or [],
                "explanation": ce.get("explanation") or "",
                "problems": ce.get("problems") or [],
            })
    else:
        judge_errors.append("faithfulness:no_claim_evaluations")
        for c in claims:
            claim_evals.append({
                "case_id": case_id,
                "mode": mode,
                "claim_id": c["claim_id"],
                "claim_type": c["claim_type"],
                "claim_text": c["claim_text"],
                "citations": c["citations"],
                "source_item_path": c["source_item_path"],
                "support_status": None,
                "support_score": None,
                "citation_supports_claim": None,
                "uses_valid_citation": None,
                "supporting_source_ids": [],
                "explanation": "judge_failed_to_return_evaluation",
                "problems": [],
            })

    citation_evals: List[Dict[str, Any]] = []
    cit_eval_meta: Optional[Dict[str, Any]] = None
    if mode == "rag" and citation_runner is not None and claims:
        c_eval = run_citation_accuracy_judge(
            citation_runner,
            case_id=case_id,
            patient_case=patient_case,
            claims=claims,
            retrieved_chunks=reference_chunks,
            reference_context=reference_context,
        )
        timings["citation_judge_sec"] = c_eval.get("elapsed_sec") or 0.0
        cit_eval_meta = {
            "errors": c_eval.get("errors") or [],
            "attempts": c_eval.get("attempts"),
            "mock": c_eval.get("mock"),
        }
        if c_eval.get("errors"):
            judge_errors.extend([f"citation:{e}" for e in c_eval["errors"]])
        cev_parsed = c_eval.get("parsed") or {}
        ce_raw = cev_parsed.get("citation_evaluations") if isinstance(cev_parsed, dict) else None
        if isinstance(ce_raw, list):
            for ce in ce_raw:
                if isinstance(ce, dict):
                    citation_evals.append(ce)
        else:
            judge_errors.append("citation:no_citation_evaluations")

    relevance_parsed: Optional[Dict[str, Any]] = None
    if relevance_runner is not None and valid_json:
        r_eval = run_answer_relevance_judge(
            relevance_runner,
            case_id=case_id,
            mode=mode,
            patient_case=patient_case,
            answer_json=answer_json,
        )
        timings["relevance_judge_sec"] = r_eval.get("elapsed_sec") or 0.0
        if r_eval.get("errors"):
            judge_errors.extend([f"relevance:{e}" for e in r_eval["errors"]])
        if isinstance(r_eval.get("parsed"), dict):
            relevance_parsed = r_eval["parsed"]
        else:
            judge_errors.append("relevance:invalid_payload")

    per_case = aggregate_per_case(
        case_id=case_id,
        mode=mode,
        claim_evals=claim_evals,
    )

    citation_metrics: Optional[Dict[str, Any]] = None
    if mode == "rag":
        citation_metrics = compute_citation_metrics_for_rag(
            answer_json=answer_json if valid_json else None,
            retrieved_chunks=reference_chunks,
            claims=claims,
            claim_evals=claim_evals,
            citation_evals=citation_evals,
        )

    diag_match = match_diagnoses(expected_diagnoses, answer_json if valid_json else None)

    for ce in claim_evals:
        _write_record(fh_claims, ce)

    has_none_status = any(ce.get("support_status") is None for ce in claim_evals)
    judge_failed = (
        not isinstance(f_parsed, dict)
        or not isinstance(claim_evals_raw, list)
        or has_none_status
    )
    if judge_failed:
        _write_record(fh_failed, {
            "case_id": case_id,
            "mode": mode,
            "errors": judge_errors,
            "claims_total": len(claims),
            "claims_with_none_status": sum(
                1 for ce in claim_evals if ce.get("support_status") is None
            ),
            "judge_attempts": (f_eval.get("attempts") if isinstance(f_eval, dict) else None),
            "raw_text": (f_eval.get("raw_text") or "")[:4000],
            "created_at": _utc_now(),
        })

    return {
        "case_id": case_id,
        "mode": mode,
        "valid_json": valid_json,
        "num_claims": len(claims),
        "per_case": per_case,
        "citation_metrics": citation_metrics,
        "relevance": relevance_parsed,
        "diagnosis_match": diag_match,
        "faithfulness_overall": (f_parsed.get("overall") if isinstance(f_parsed, dict) else None),
        "judge_errors": judge_errors,
        "judge_timings_sec": timings,
        "created_at": _utc_now(),
        "expected_diagnoses": list(expected_diagnoses),
    }

SUMMARY_COLUMNS: List[str] = [
    "case_id",
    "difficulty",
    "case_type",
    "rag_valid_json",
    "no_rag_valid_json",
    "rag_num_claims",
    "no_rag_num_claims",
    "rag_supported_claims",
    "no_rag_supported_claims",
    "rag_partially_supported_claims",
    "no_rag_partially_supported_claims",
    "rag_unsupported_claims",
    "no_rag_unsupported_claims",
    "rag_contradicted_claims",
    "no_rag_contradicted_claims",
    "rag_faithfulness_strict",
    "no_rag_faithfulness_strict",
    "rag_faithfulness_soft",
    "no_rag_faithfulness_soft",
    "rag_hallucination_rate",
    "no_rag_hallucination_rate",
    "rag_citation_coverage_claim_level",
    "rag_citation_coverage_item_level",
    "rag_citation_validity_rate",
    "rag_citation_accuracy_rate",
    "rag_citation_accuracy_source",
    "rag_answer_relevance_score",
    "no_rag_answer_relevance_score",
    "rag_clinical_usefulness_score",
    "no_rag_clinical_usefulness_score",
    "rag_safety_score",
    "no_rag_safety_score",
    "faithfulness_improvement_abs",
    "hallucination_reduction_abs",
    "rag_top1_diagnosis_match",
    "no_rag_top1_diagnosis_match",
    "rag_top3_diagnosis_match",
    "no_rag_top3_diagnosis_match",
]


def _summary_row(
    case_meta: Dict[str, Any],
    rag_eval: Optional[Dict[str, Any]],
    no_rag_eval: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    rag = rag_eval or {}
    no_rag = no_rag_eval or {}
    rpc = rag.get("per_case") or {}
    npc = no_rag.get("per_case") or {}
    rcm = rag.get("citation_metrics") or {}
    rrel = rag.get("relevance") or {}
    nrel = no_rag.get("relevance") or {}
    rdm = rag.get("diagnosis_match") or {}
    ndm = no_rag.get("diagnosis_match") or {}

    def _delta(a: Any, b: Any) -> Optional[float]:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(a - b, 4)
        return None

    return {
        "case_id": case_meta.get("case_id"),
        "difficulty": case_meta.get("difficulty"),
        "case_type": case_meta.get("case_type"),
        "rag_valid_json": rag.get("valid_json"),
        "no_rag_valid_json": no_rag.get("valid_json"),
        "rag_num_claims": rpc.get("num_claims"),
        "no_rag_num_claims": npc.get("num_claims"),
        "rag_supported_claims": rpc.get("supported_claims"),
        "no_rag_supported_claims": npc.get("supported_claims"),
        "rag_partially_supported_claims": rpc.get("partially_supported_claims"),
        "no_rag_partially_supported_claims": npc.get("partially_supported_claims"),
        "rag_unsupported_claims": rpc.get("unsupported_claims"),
        "no_rag_unsupported_claims": npc.get("unsupported_claims"),
        "rag_contradicted_claims": rpc.get("contradicted_claims"),
        "no_rag_contradicted_claims": npc.get("contradicted_claims"),
        "rag_faithfulness_strict": rpc.get("faithfulness_strict"),
        "no_rag_faithfulness_strict": npc.get("faithfulness_strict"),
        "rag_faithfulness_soft": rpc.get("faithfulness_soft"),
        "no_rag_faithfulness_soft": npc.get("faithfulness_soft"),
        "rag_hallucination_rate": rpc.get("hallucination_rate"),
        "no_rag_hallucination_rate": npc.get("hallucination_rate"),
        "rag_citation_coverage_claim_level": rcm.get("citation_coverage_claim_level"),
        "rag_citation_coverage_item_level": rcm.get("citation_coverage_item_level"),
        "rag_citation_validity_rate": rcm.get("citation_validity_rate"),
        "rag_citation_accuracy_rate": rcm.get("citation_accuracy_rate"),
        "rag_citation_accuracy_source": rcm.get("citation_accuracy_source"),
        "rag_answer_relevance_score": rrel.get("answer_relevance_score"),
        "no_rag_answer_relevance_score": nrel.get("answer_relevance_score"),
        "rag_clinical_usefulness_score": rrel.get("clinical_usefulness_score"),
        "no_rag_clinical_usefulness_score": nrel.get("clinical_usefulness_score"),
        "rag_safety_score": rrel.get("safety_score"),
        "no_rag_safety_score": nrel.get("safety_score"),
        "faithfulness_improvement_abs": _delta(
            rpc.get("faithfulness_soft"), npc.get("faithfulness_soft")
        ),
        "hallucination_reduction_abs": _delta(
            npc.get("hallucination_rate"), rpc.get("hallucination_rate")
        ),
        "rag_top1_diagnosis_match": rdm.get("top1_match"),
        "no_rag_top1_diagnosis_match": ndm.get("top1_match"),
        "rag_top3_diagnosis_match": rdm.get("top3_match"),
        "no_rag_top3_diagnosis_match": ndm.get("top3_match"),
    }


def _write_summary_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in SUMMARY_COLUMNS})

def run_llm_evaluation(
    *,
    pairs_path: Optional[str | Path],
    rag_answers_path: Optional[str | Path],
    no_rag_answers_path: Optional[str | Path],
    clinical_cases_path: Optional[str | Path],
    llm_config: str | Path,
    judge_prompt_config: str | Path,
    citation_judge_prompt_config: Optional[str | Path],
    relevance_judge_prompt_config: Optional[str | Path],
    output_dir: str | Path,
    limit: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = False,
    mode: str = "both",
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    max_claims_per_answer: int = 20,
    judge_max_retries: int = 1,
) -> Dict[str, Any]:
    if mode not in ("both", "rag", "no_rag"):
        raise ValueError(f"mode must be one of both/rag/no_rag, got {mode!r}")

    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    pairs_p = Path(pairs_path) if pairs_path else None
    rag_p = Path(rag_answers_path) if rag_answers_path else None
    no_rag_p = Path(no_rag_answers_path) if no_rag_answers_path else None
    cases_p = Path(clinical_cases_path) if clinical_cases_path else None

    if pairs_p and (rag_p or no_rag_p):
        raise ValueError("Specify either --pairs-path OR --rag-answers-path/--no-rag-answers-path, not both.")

    pairs = _load_pairs(pairs_path=pairs_p, rag_path=rag_p, no_rag_path=no_rag_p)
    if not pairs:
        raise RuntimeError("No pairs loaded — check --pairs-path / --rag-answers-path / --no-rag-answers-path.")
    cases_meta = _load_clinical_cases(cases_p)

    if limit is not None and limit > 0:
        pairs = pairs[:limit]

    llm_client = LLMClient(str(llm_config), provider=judge_provider, model_name=judge_model)
    logger.info("Judge LLM: provider=%s type=%s model=%s",
                llm_client.provider_key, llm_client.provider_type, llm_client.model_name)

    f_prompt = load_prompt_config(judge_prompt_config)
    f_runner = JudgeRunner(llm_client, f_prompt, max_retries=judge_max_retries)

    citation_runner: Optional[JudgeRunner] = None
    if citation_judge_prompt_config:
        c_prompt = load_prompt_config(citation_judge_prompt_config)
        citation_runner = JudgeRunner(llm_client, c_prompt, max_retries=judge_max_retries)

    relevance_runner: Optional[JudgeRunner] = None
    if relevance_judge_prompt_config:
        r_prompt = load_prompt_config(relevance_judge_prompt_config)
        relevance_runner = JudgeRunner(llm_client, r_prompt, max_retries=judge_max_retries)

    claims_path = out_dir / "claim_evaluations.jsonl"
    cases_path_out = out_dir / "case_evaluations.jsonl"
    summary_csv = out_dir / "rag_vs_no_rag_eval_summary.csv"
    summary_json = out_dir / "rag_vs_no_rag_eval_summary.json"
    metrics_json = out_dir / "llm_eval_metrics.json"
    metrics_csv = out_dir / "llm_eval_metrics.csv"
    failed_path = out_dir / "failed_judge_cases.jsonl"

    seen: Set[Tuple[str, str]] = set()
    if resume:
        seen = _existing_case_mode_keys(cases_path_out)
        if seen:
            logger.info("--resume: already evaluated %d (case,mode) pairs", len(seen))

    fh_claims = _open_append(claims_path, overwrite=overwrite, resume=resume)
    fh_cases = _open_append(cases_path_out, overwrite=overwrite, resume=resume)
    fh_failed = _open_append(failed_path, overwrite=overwrite, resume=resume)

    case_evaluations_all: List[Dict[str, Any]] = []
    if resume and cases_path_out.exists():
        try:
            case_evaluations_all = list(read_jsonl_list(cases_path_out))
        except Exception as e:
            logger.warning("Could not load existing case_evaluations.jsonl on --resume: %s", e)
            case_evaluations_all = []

    started_at = _utc_now()
    t_start = time.perf_counter()

    try:
        for pair in tqdm(pairs, desc="llm-eval", file=sys.stdout, dynamic_ncols=True):
            case_id = str(pair.get("case_id") or "")
            if not case_id:
                logger.warning("Skipping pair without case_id: %s", pair)
                continue

            patient_case = (pair.get("patient_case") or "").strip()
            rag_record = pair.get("rag") or {}
            no_rag_record = pair.get("no_rag") or {}

            if not patient_case:
                pivot = rag_record or no_rag_record or {}
                patient_case = (pivot.get("patient_case") or "").strip()

            reference_chunks = list((rag_record or {}).get("retrieved_chunks") or [])
            reference_context = format_reference_context(reference_chunks)

            case_meta = cases_meta.get(case_id) or {}
            expected_diagnoses = case_meta.get("expected_diagnoses") or []

            if mode in ("both", "rag") and rag_record and (case_id, "rag") not in seen:
                try:
                    rag_eval = _evaluate_one_mode(
                        case_id=case_id,
                        mode="rag",
                        patient_case=patient_case,
                        answer_record=rag_record,
                        reference_chunks=reference_chunks,
                        reference_context=reference_context,
                        faithfulness_runner=f_runner,
                        citation_runner=citation_runner,
                        relevance_runner=relevance_runner,
                        expected_diagnoses=expected_diagnoses,
                        max_claims_per_answer=max_claims_per_answer,
                        fh_claims=fh_claims,
                        fh_failed=fh_failed,
                    )
                    _write_record(fh_cases, rag_eval)
                    case_evaluations_all.append(rag_eval)
                except Exception as e:
                    logger.exception("RAG eval failed for case %s", case_id)
                    _write_record(fh_failed, {
                        "case_id": case_id, "mode": "rag",
                        "errors": [f"unexpected:{e}"],
                        "created_at": _utc_now(),
                    })

            if mode in ("both", "no_rag") and no_rag_record and (case_id, "no_rag") not in seen:
                try:
                    nr_eval = _evaluate_one_mode(
                        case_id=case_id,
                        mode="no_rag",
                        patient_case=patient_case,
                        answer_record=no_rag_record,
                        reference_chunks=reference_chunks,
                        reference_context=reference_context,
                        faithfulness_runner=f_runner,
                        citation_runner=None,
                        relevance_runner=relevance_runner,
                        expected_diagnoses=expected_diagnoses,
                        max_claims_per_answer=max_claims_per_answer,
                        fh_claims=fh_claims,
                        fh_failed=fh_failed,
                    )
                    _write_record(fh_cases, nr_eval)
                    case_evaluations_all.append(nr_eval)
                except Exception as e: 
                    logger.exception("no-RAG eval failed for case %s", case_id)
                    _write_record(fh_failed, {
                        "case_id": case_id, "mode": "no_rag",
                        "errors": [f"unexpected:{e}"],
                        "created_at": _utc_now(),
                    })
    finally:
        fh_claims.close()
        fh_cases.close()
        fh_failed.close()

    elapsed = time.perf_counter() - t_start
    finished_at = _utc_now()

    global_metrics = aggregate_global(case_evaluations_all)
    global_metrics["meta"] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": round(elapsed, 4),
        "judge_provider": llm_client.provider_key,
        "judge_provider_type": llm_client.provider_type,
        "judge_model": llm_client.model_name,
        "judge_prompt_config": str(judge_prompt_config),
        "citation_judge_prompt_config": str(citation_judge_prompt_config) if citation_judge_prompt_config else None,
        "relevance_judge_prompt_config": str(relevance_judge_prompt_config) if relevance_judge_prompt_config else None,
        "pairs_path": str(pairs_p) if pairs_p else None,
        "rag_answers_path": str(rag_p) if rag_p else None,
        "no_rag_answers_path": str(no_rag_p) if no_rag_p else None,
        "clinical_cases_path": str(cases_p) if cases_p else None,
        "limit": limit,
        "mode": mode,
        "max_claims_per_answer": max_claims_per_answer,
        "num_pairs_input": len(pairs),
        "num_case_evaluations_total": len(case_evaluations_all),
    }

    write_json(metrics_json, global_metrics)

    metrics_row = _flatten_metrics_row(global_metrics)
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in metrics_row.items()})

    by_id_eval: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for ce in case_evaluations_all:
        cid = ce.get("case_id")
        m = ce.get("mode")
        if not cid or not m:
            continue
        by_id_eval.setdefault(cid, {})[m] = ce

    summary_rows: List[Dict[str, Any]] = []
    for pair in pairs:
        cid = str(pair.get("case_id") or "")
        if not cid:
            continue
        meta = cases_meta.get(cid) or {}
        rag_e = by_id_eval.get(cid, {}).get("rag")
        nr_e = by_id_eval.get(cid, {}).get("no_rag")
        summary_rows.append(_summary_row(
            case_meta={"case_id": cid, "difficulty": meta.get("difficulty"), "case_type": meta.get("case_type")},
            rag_eval=rag_e,
            no_rag_eval=nr_e,
        ))
    _write_summary_csv(summary_rows, summary_csv)
    write_json(summary_json, {"rows": summary_rows, "num_rows": len(summary_rows)})

    try:
        from .build_llm_eval_report import build_report
        report_path = out_dir / "llm_eval_report.md"
        build_report(
            output_dir=out_dir,
            metrics=global_metrics,
            summary_rows=summary_rows,
            case_evaluations=case_evaluations_all,
            cases_meta=cases_meta,
            failed_path=failed_path,
            judge_prompt_config_path=str(judge_prompt_config),
            citation_judge_prompt_config_path=str(citation_judge_prompt_config) if citation_judge_prompt_config else None,
            relevance_judge_prompt_config_path=str(relevance_judge_prompt_config) if relevance_judge_prompt_config else None,
            output_path=report_path,
        )
    except Exception:
        logger.exception("build_llm_eval_report failed; metrics still saved")

    return {
        "output_dir": str(out_dir),
        "num_pairs": len(pairs),
        "num_case_evaluations": len(case_evaluations_all),
        "metrics_json": str(metrics_json),
        "metrics_csv": str(metrics_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "claim_evaluations": str(claims_path),
        "case_evaluations": str(cases_path_out),
        "failed_judge_cases": str(failed_path),
        "report_md": str(out_dir / "llm_eval_report.md"),
        "elapsed_sec": round(elapsed, 4),
    }


def _flatten_metrics_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    rag = metrics.get("rag") or {}
    no_rag = metrics.get("no_rag") or {}
    cmp_ = metrics.get("comparison") or {}
    targets = metrics.get("targets") or {}

    def _pull(d: Dict[str, Any], keys: Sequence[str], prefix: str) -> Dict[str, Any]:
        return {f"{prefix}_{k}": d.get(k) for k in keys}

    rag_keys = [
        "num_cases", "valid_json_count", "valid_json_rate",
        "total_claims", "total_supported_claims", "total_partially_supported_claims",
        "total_unsupported_claims", "total_contradicted_claims",
        "faithfulness_strict", "faithfulness_soft", "hallucination_rate",
        "faithfulness_strict_micro", "faithfulness_soft_micro", "hallucination_rate_micro",
        "citation_coverage_item_level", "citation_coverage_claim_level",
        "citation_validity_rate", "citation_accuracy_rate", "invalid_citation_count_total",
        "answer_relevance_avg", "clinical_usefulness_avg", "safety_avg",
        "states_final_diagnosis_count", "has_disclaimer_count",
        "top1_diagnosis_match_rate", "top3_diagnosis_match_rate",
        "diagnosis_coverage_avg", "diagnosis_match_evaluated_cases",
    ]
    no_rag_keys = [
        "num_cases", "valid_json_count", "valid_json_rate",
        "total_claims", "total_supported_claims", "total_partially_supported_claims",
        "total_unsupported_claims", "total_contradicted_claims",
        "faithfulness_strict", "faithfulness_soft", "hallucination_rate",
        "faithfulness_strict_micro", "faithfulness_soft_micro", "hallucination_rate_micro",
        "answer_relevance_avg", "clinical_usefulness_avg", "safety_avg",
        "states_final_diagnosis_count", "has_disclaimer_count",
        "top1_diagnosis_match_rate", "top3_diagnosis_match_rate",
    ]
    cmp_keys = [
        "faithfulness_soft_improvement_abs", "faithfulness_soft_improvement_rel",
        "hallucination_rate_reduction_abs",
        "rag_meets_faithfulness_target",
        "rag_meets_hallucination_target",
        "rag_meets_citation_accuracy_target",
        "rag_meets_faithfulness_improvement_target",
        "rag_meets_hallucination_reduction_target",
        "case_level_wins_rag_better",
        "case_level_losses_rag_worse",
        "case_level_ties",
    ]
    out: Dict[str, Any] = {"num_cases_total": metrics.get("num_cases")}
    out.update(_pull(rag, rag_keys, "rag"))
    out.update(_pull(no_rag, no_rag_keys, "no_rag"))
    out.update(_pull(cmp_, cmp_keys, ""))
    out.update({f"target_{k}": v for k, v in targets.items()})
    return out

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-a-judge evaluator for RAG vs no-RAG answers")
    p.add_argument("--pairs-path", default=None, type=str)
    p.add_argument("--rag-answers-path", default=None, type=str)
    p.add_argument("--no-rag-answers-path", default=None, type=str)
    p.add_argument("--clinical-cases-path", default=None, type=str,
                   help="Optional JSONL c expected_diagnoses / difficulty / case_type.")
    p.add_argument("--llm-config", required=True, type=str)
    p.add_argument("--judge-prompt-config", required=True, type=str)
    p.add_argument("--citation-judge-prompt-config", default=None, type=str)
    p.add_argument("--relevance-judge-prompt-config", default=None, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mode", choices=["both", "rag", "no_rag"], default="both")
    p.add_argument("--judge-provider", default=None, type=str,
                   help="Override llm.provider (например, mock для smoke).")
    p.add_argument("--judge-model", default=None, type=str)
    p.add_argument("--max-claims-per-answer", type=int, default=20)
    p.add_argument("--judge-max-retries", type=int, default=1)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    setup_logging(
        verbose=args.verbose,
        log_file=out_dir / "evaluate_llm_answers.log",
        name="llm-eval",
    )
    logger.info("Args: %s", vars(args))

    try:
        info = run_llm_evaluation(
            pairs_path=args.pairs_path,
            rag_answers_path=args.rag_answers_path,
            no_rag_answers_path=args.no_rag_answers_path,
            clinical_cases_path=args.clinical_cases_path,
            llm_config=args.llm_config,
            judge_prompt_config=args.judge_prompt_config,
            citation_judge_prompt_config=args.citation_judge_prompt_config,
            relevance_judge_prompt_config=args.relevance_judge_prompt_config,
            output_dir=out_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            resume=args.resume,
            mode=args.mode,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            max_claims_per_answer=args.max_claims_per_answer,
            judge_max_retries=args.judge_max_retries,
        )
        logger.info("LLM evaluation done: %s", info)
    except FileExistsError as e:
        logger.error("%s", e)
        return 2
    except (ValueError, RuntimeError) as e:
        logger.error("%s", e)
        return 2
    except Exception: 
        logger.exception("LLM evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
