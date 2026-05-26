from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import ensure_dir, read_json, read_jsonl_list, write_json

logger = logging.getLogger(__name__)


def _safe_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def _safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return read_jsonl_list(path)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return []


def _safe_read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e: 
        logger.warning("Could not read %s: %s", path, e)
        return []


def _avg(values: List[float]) -> Optional[float]:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    return float(sum(cleaned) / len(cleaned)) if cleaned else None


def collect_retrieval(out_dir: Path) -> Dict[str, Any]:
    reports_dir = out_dir / "retrieval" / "reports"
    best = _safe_read_json(reports_dir / "best_embedding_model.json")
    comp_json = _safe_read_json(reports_dir / "embedding_model_comparison.json")

    info: Dict[str, Any] = {"enabled": True, "status": "unknown"}
    if not best and not comp_json:
        return {"enabled": False, "status": "skipped"}

    if best:
        info["best_embedding_model"] = best.get("best_model_key")
        m = best.get("metrics") or {}
        info["best_recall_at_5"] = m.get("recall_at_5")
        info["best_precision_at_5"] = m.get("precision_at_5")
        info["best_mrr"] = m.get("mrr")
        info["best_hit_at_5"] = m.get("hit_at_5")

    if comp_json:
        info["per_model"] = comp_json
        info["num_models"] = len(comp_json) if isinstance(comp_json, list) else None

    info["status"] = "completed" if best or comp_json else "unknown"
    return info


def collect_reranking(out_dir: Path) -> Dict[str, Any]:
    rerank_json_path = out_dir / "rerank" / "rerank_comparison_metrics.json"
    data = _safe_read_json(rerank_json_path)
    if not data:
        return {"enabled": False, "status": "skipped"}

    cfg = (data or {}).get("config") or {}
    final_m = (data or {}).get("final_metrics") or {}
    dense_m = (data or {}).get("dense_metrics") or {}
    rer_m = (data or {}).get("reranked_metrics") or {}

    return {
        "enabled": True,
        "status": "completed",
        "embedding_model": cfg.get("embedding_model_key"),
        "reranker": cfg.get("reranker_key"),
        "context_selection": cfg.get("context_selection"),
        "candidate_top_k": cfg.get("candidate_top_k"),
        "final_top_k": cfg.get("final_top_k"),
        "dense_document_recall_at_5": dense_m.get("document_recall_at_5"),
        "dense_document_precision_at_5": dense_m.get("document_precision_at_5"),
        "reranked_document_recall_at_5": rer_m.get("document_recall_at_5"),
        "reranked_document_precision_at_5": rer_m.get("document_precision_at_5"),
        "final_document_recall_at_5": final_m.get("document_recall_at_5"),
        "final_document_precision_at_5": final_m.get("document_precision_at_5"),
        "final_chunk_hit_at_5": final_m.get("chunk_hit_at_5"),
        "final_section_hit_at_5": final_m.get("section_hit_at_5"),
        "final_page_hit_at_5": final_m.get("page_hit_at_5"),
    }


def _generation_metrics_for(records: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    if not records:
        return {
            "mode": mode,
            "num_cases": 0,
            "num_success": 0,
            "num_failed": 0,
            "valid_json_count": 0,
            "valid_json_rate": None,
            "avg_generation_time_sec": None,
            "avg_total_time_sec": None,
            "avg_prompt_tokens": None,
            "avg_completion_tokens": None,
            "avg_total_tokens": None,
            "avg_citation_count": None,
            "invalid_citations_total": 0,
            "avg_citation_coverage_estimate": None,
        }
    n = len(records)
    valid = [r for r in records if isinstance(r.get("answer_json"), dict)]
    n_valid = len(valid)

    gen_times = [(r.get("timing") or {}).get("generation_time_sec") for r in records]
    total_times = [(r.get("timing") or {}).get("total_time_sec") for r in records]
    p_tok = [(r.get("llm") or {}).get("usage", {}).get("prompt_tokens") for r in records if isinstance(r.get("llm"), dict)]
    c_tok = [(r.get("llm") or {}).get("usage", {}).get("completion_tokens") for r in records if isinstance(r.get("llm"), dict)]
    t_tok = [(r.get("llm") or {}).get("usage", {}).get("total_tokens") for r in records if isinstance(r.get("llm"), dict)]

    cit_counts = [(r.get("citation_validation") or {}).get("citation_count", 0) for r in records]
    invalid_cit = [(r.get("citation_validation") or {}).get("invalid_citation_count", 0) for r in records]
    coverage = [(r.get("citation_validation") or {}).get("citation_coverage_estimate") for r in records]

    return {
        "mode": mode,
        "num_cases": n,
        "num_success": n_valid,
        "num_failed": n - n_valid,
        "valid_json_count": n_valid,
        "valid_json_rate": round(n_valid / n, 4) if n else None,
        "avg_generation_time_sec": _round(_avg([v for v in gen_times if isinstance(v, (int, float))])),
        "avg_total_time_sec": _round(_avg([v for v in total_times if isinstance(v, (int, float))])),
        "avg_prompt_tokens": _round(_avg([v for v in p_tok if isinstance(v, (int, float))])),
        "avg_completion_tokens": _round(_avg([v for v in c_tok if isinstance(v, (int, float))])),
        "avg_total_tokens": _round(_avg([v for v in t_tok if isinstance(v, (int, float))])),
        "avg_citation_count": _round(_avg([float(v) for v in cit_counts if isinstance(v, (int, float))])),
        "invalid_citations_total": int(sum(v for v in invalid_cit if isinstance(v, (int, float)))),
        "avg_citation_coverage_estimate": _round(_avg([v for v in coverage if isinstance(v, (int, float))])),
    }


def _round(v: Optional[float], n: int = 4) -> Optional[float]:
    return round(v, n) if isinstance(v, (int, float)) else None


def collect_generation(out_dir: Path) -> Dict[str, Any]:
    rag_path = out_dir / "generation" / "rag_answers.jsonl"
    no_rag_path = out_dir / "generation" / "no_rag_answers.jsonl"
    rag_records = _safe_read_jsonl(rag_path)
    no_rag_records = _safe_read_jsonl(no_rag_path)

    rag_metrics = _generation_metrics_for(rag_records, "rag") if rag_records else None
    no_rag_metrics = _generation_metrics_for(no_rag_records, "no_rag") if no_rag_records else None

    info: Dict[str, Any] = {
        "enabled": bool(rag_metrics or no_rag_metrics),
        "status": "completed" if (rag_metrics or no_rag_metrics) else "skipped",
        "rag": rag_metrics,
        "no_rag": no_rag_metrics,
    }
    if rag_metrics:
        info["num_rag_answers"] = rag_metrics["num_cases"]
        info["rag_valid_json_rate"] = rag_metrics["valid_json_rate"]
        info["rag_avg_citation_count"] = rag_metrics["avg_citation_count"]
        info["rag_avg_citation_coverage_estimate"] = rag_metrics["avg_citation_coverage_estimate"]
        info["rag_invalid_citations_total"] = rag_metrics["invalid_citations_total"]
    if no_rag_metrics:
        info["num_no_rag_answers"] = no_rag_metrics["num_cases"]
        info["no_rag_valid_json_rate"] = no_rag_metrics["valid_json_rate"]
    if rag_records:
        info["llm_provider"] = (rag_records[0].get("llm") or {}).get("provider")
        info["llm_model_name"] = (rag_records[0].get("llm") or {}).get("model_name")
    elif no_rag_records:
        info["llm_provider"] = (no_rag_records[0].get("llm") or {}).get("provider")
        info["llm_model_name"] = (no_rag_records[0].get("llm") or {}).get("model_name")
    return info


def collect_comparison(out_dir: Path) -> Dict[str, Any]:
    pairs = out_dir / "generation" / "rag_vs_no_rag_pairs.jsonl"
    summary = out_dir / "generation" / "rag_vs_no_rag_summary.csv"
    if not pairs.exists() and not summary.exists():
        return {"enabled": False, "status": "skipped"}
    rows = _safe_read_csv(summary)
    return {
        "enabled": True,
        "status": "completed",
        "num_cases": len(rows),
        "files": {"pairs": str(pairs), "summary_csv": str(summary)},
    }


def collect_llm_eval(out_dir: Path) -> Dict[str, Any]:
    metrics_path = out_dir / "llm_eval" / "llm_eval_metrics.json"
    payload = _safe_read_json(metrics_path)
    if not payload:
        return {"enabled": False, "status": "skipped"}

    rag = payload.get("rag") or {}
    no_rag = payload.get("no_rag") or {}
    cmp_ = payload.get("comparison") or {}
    meta = payload.get("meta") or {}
    targets = payload.get("targets") or {}

    return {
        "enabled": True,
        "status": "completed",
        "num_cases": payload.get("num_cases"),
        "judge_provider": meta.get("judge_provider"),
        "judge_provider_type": meta.get("judge_provider_type"),
        "judge_model": meta.get("judge_model"),
        "judge_prompt_config": meta.get("judge_prompt_config"),
        "citation_judge_prompt_config": meta.get("citation_judge_prompt_config"),
        "relevance_judge_prompt_config": meta.get("relevance_judge_prompt_config"),
        "rag": rag,
        "no_rag": no_rag,
        "comparison": cmp_,
        "targets": targets,
        "rag_faithfulness_strict": rag.get("faithfulness_strict"),
        "rag_faithfulness_soft": rag.get("faithfulness_soft"),
        "no_rag_faithfulness_strict": no_rag.get("faithfulness_strict"),
        "no_rag_faithfulness_soft": no_rag.get("faithfulness_soft"),
        "rag_hallucination_rate": rag.get("hallucination_rate"),
        "no_rag_hallucination_rate": no_rag.get("hallucination_rate"),
        "rag_citation_coverage_claim_level": rag.get("citation_coverage_claim_level"),
        "rag_citation_coverage_item_level": rag.get("citation_coverage_item_level"),
        "rag_citation_validity_rate": rag.get("citation_validity_rate"),
        "rag_citation_accuracy_rate": rag.get("citation_accuracy_rate"),
        "rag_answer_relevance_avg": rag.get("answer_relevance_avg"),
        "no_rag_answer_relevance_avg": no_rag.get("answer_relevance_avg"),
        "rag_clinical_usefulness_avg": rag.get("clinical_usefulness_avg"),
        "no_rag_clinical_usefulness_avg": no_rag.get("clinical_usefulness_avg"),
        "rag_safety_avg": rag.get("safety_avg"),
        "no_rag_safety_avg": no_rag.get("safety_avg"),
        "faithfulness_soft_improvement_abs": cmp_.get("faithfulness_soft_improvement_abs"),
        "faithfulness_soft_improvement_rel": cmp_.get("faithfulness_soft_improvement_rel"),
        "hallucination_rate_reduction_abs": cmp_.get("hallucination_rate_reduction_abs"),
        "case_level_wins_rag_better": cmp_.get("case_level_wins_rag_better"),
        "case_level_losses_rag_worse": cmp_.get("case_level_losses_rag_worse"),
        "case_level_ties": cmp_.get("case_level_ties"),
        "targets_met_faithfulness": cmp_.get("rag_meets_faithfulness_target"),
        "targets_met_hallucination": cmp_.get("rag_meets_hallucination_target"),
        "targets_met_citation_accuracy": cmp_.get("rag_meets_citation_accuracy_target"),
        "targets_met_faithfulness_improvement": cmp_.get("rag_meets_faithfulness_improvement_target"),
        "targets_met_hallucination_reduction": cmp_.get("rag_meets_hallucination_reduction_target"),
    }


def build_final_metrics(
    out_dir: Path,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    retrieval: Optional[Dict[str, Any]] = None,
    reranking: Optional[Dict[str, Any]] = None,
    generation: Optional[Dict[str, Any]] = None,
    llm_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rer = reranking or {}
    gen = generation or {}
    rag_m = (gen.get("rag") or {}) if gen else {}
    no_rag_m = (gen.get("no_rag") or {}) if gen else {}
    le = llm_eval or {}

    cfg = cfg or {}
    cfg_data = (cfg.get("data") or {})
    cfg_gen = (cfg.get("generation") or {})
    cfg_le = (cfg.get("llm_evaluation") or {})

    return {
        "embedding_model": rer.get("embedding_model"),
        "reranker": rer.get("reranker"),
        "context_selection": rer.get("context_selection"),
        "retrieval_dataset": cfg_data.get("retrieval_queries_path"),
        "number_of_retrieval_queries": _count_jsonl(out_dir / "rerank" / "detailed_results.jsonl") or _count_jsonl(_first_existing(out_dir / "retrieval" / "retrieval_results")),
        "dense_document_recall_at_5": rer.get("dense_document_recall_at_5"),
        "dense_document_precision_at_5": rer.get("dense_document_precision_at_5"),
        "reranked_document_recall_at_5": rer.get("reranked_document_recall_at_5"),
        "reranked_document_precision_at_5": rer.get("reranked_document_precision_at_5"),
        "final_document_recall_at_5": rer.get("final_document_recall_at_5"),
        "final_document_precision_at_5": rer.get("final_document_precision_at_5"),
        "final_chunk_hit_at_5": rer.get("final_chunk_hit_at_5"),
        "final_page_hit_at_5": rer.get("final_page_hit_at_5"),
        "final_section_hit_at_5": rer.get("final_section_hit_at_5"),
        "llm_provider": gen.get("llm_provider"),
        "llm_model_name": gen.get("llm_model_name"),
        "prompt_config_rag": cfg_gen.get("prompt_config_rag"),
        "prompt_config_no_rag": cfg_gen.get("prompt_config_no_rag"),
        "clinical_cases_path": cfg_data.get("clinical_cases_path"),
        "generation_limit": cfg_gen.get("limit"),
        "rag_num_answers": rag_m.get("num_cases") if rag_m else None,
        "no_rag_num_answers": no_rag_m.get("num_cases") if no_rag_m else None,
        "rag_valid_json_rate": rag_m.get("valid_json_rate") if rag_m else None,
        "no_rag_valid_json_rate": no_rag_m.get("valid_json_rate") if no_rag_m else None,
        "rag_avg_citation_count": rag_m.get("avg_citation_count") if rag_m else None,
        "rag_invalid_citations_total": rag_m.get("invalid_citations_total") if rag_m else None,
        "rag_avg_citation_coverage": rag_m.get("avg_citation_coverage_estimate") if rag_m else None,
        "llm_eval_enabled": le.get("enabled"),
        "judge_provider": le.get("judge_provider"),
        "judge_model": le.get("judge_model"),
        "judge_prompt_config": cfg_le.get("judge_prompt_config") or le.get("judge_prompt_config"),
        "citation_judge_prompt_config": cfg_le.get("citation_judge_prompt_config") or le.get("citation_judge_prompt_config"),
        "relevance_judge_prompt_config": cfg_le.get("relevance_judge_prompt_config") or le.get("relevance_judge_prompt_config"),
        "rag_faithfulness_strict": le.get("rag_faithfulness_strict"),
        "rag_faithfulness_soft": le.get("rag_faithfulness_soft"),
        "no_rag_faithfulness_strict": le.get("no_rag_faithfulness_strict"),
        "no_rag_faithfulness_soft": le.get("no_rag_faithfulness_soft"),
        "rag_hallucination_rate": le.get("rag_hallucination_rate"),
        "no_rag_hallucination_rate": le.get("no_rag_hallucination_rate"),
        "rag_citation_coverage_claim_level": le.get("rag_citation_coverage_claim_level"),
        "rag_citation_coverage_item_level": le.get("rag_citation_coverage_item_level"),
        "rag_citation_validity_rate": le.get("rag_citation_validity_rate"),
        "rag_citation_accuracy_rate": le.get("rag_citation_accuracy_rate"),
        "rag_answer_relevance_avg": le.get("rag_answer_relevance_avg"),
        "no_rag_answer_relevance_avg": le.get("no_rag_answer_relevance_avg"),
        "rag_clinical_usefulness_avg": le.get("rag_clinical_usefulness_avg"),
        "no_rag_clinical_usefulness_avg": le.get("no_rag_clinical_usefulness_avg"),
        "rag_safety_avg": le.get("rag_safety_avg"),
        "no_rag_safety_avg": le.get("no_rag_safety_avg"),
        "faithfulness_soft_improvement_abs": le.get("faithfulness_soft_improvement_abs"),
        "faithfulness_soft_improvement_rel": le.get("faithfulness_soft_improvement_rel"),
        "hallucination_rate_reduction_abs": le.get("hallucination_rate_reduction_abs"),
        "case_level_wins_rag_better": le.get("case_level_wins_rag_better"),
        "case_level_losses_rag_worse": le.get("case_level_losses_rag_worse"),
        "case_level_ties": le.get("case_level_ties"),
        "targets_met_faithfulness": le.get("targets_met_faithfulness"),
        "targets_met_hallucination": le.get("targets_met_hallucination"),
        "targets_met_citation_accuracy": le.get("targets_met_citation_accuracy"),
        "targets_met_faithfulness_improvement": le.get("targets_met_faithfulness_improvement"),
        "targets_met_hallucination_reduction": le.get("targets_met_hallucination_reduction"),
    }


def _count_jsonl(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return None


def _first_existing(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob("*_detailed_results.jsonl"))
    return files[0] if files else None



def _write_csv_row(row: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _format_md(metrics: Dict[str, Any], *, status: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Final metrics summary\n")
    lines.append(f"- Status: **{status.get('status', 'unknown')}**")
    lines.append(f"- Started: {status.get('started_at', '—')}")
    lines.append(f"- Finished: {status.get('finished_at', '—')}")
    lines.append(f"- Total runtime, sec: {status.get('total_runtime_sec', '—')}\n")

    lines.append("## Retrieval / reranking\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    for k in (
        "embedding_model",
        "reranker",
        "context_selection",
        "retrieval_dataset",
        "number_of_retrieval_queries",
        "dense_document_recall_at_5",
        "dense_document_precision_at_5",
        "reranked_document_recall_at_5",
        "reranked_document_precision_at_5",
        "final_document_recall_at_5",
        "final_document_precision_at_5",
        "final_chunk_hit_at_5",
        "final_section_hit_at_5",
        "final_page_hit_at_5",
    ):
        lines.append(f"| {k} | {_pretty(metrics.get(k))} |")

    lines.append("\n## Generation\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    for k in (
        "llm_provider",
        "llm_model_name",
        "prompt_config_rag",
        "prompt_config_no_rag",
        "clinical_cases_path",
        "generation_limit",
        "rag_num_answers",
        "no_rag_num_answers",
        "rag_valid_json_rate",
        "no_rag_valid_json_rate",
        "rag_avg_citation_count",
        "rag_invalid_citations_total",
        "rag_avg_citation_coverage",
    ):
        lines.append(f"| {k} | {_pretty(metrics.get(k))} |")

    if metrics.get("llm_eval_enabled"):
        lines.append("\n## LLM evaluation\n")
        lines.append("| Метрика | RAG | no-RAG |")
        lines.append("|---|---|---|")
        rows = [
            ("faithfulness_strict", "rag_faithfulness_strict", "no_rag_faithfulness_strict"),
            ("faithfulness_soft", "rag_faithfulness_soft", "no_rag_faithfulness_soft"),
            ("hallucination_rate", "rag_hallucination_rate", "no_rag_hallucination_rate"),
            ("answer_relevance_avg", "rag_answer_relevance_avg", "no_rag_answer_relevance_avg"),
            ("clinical_usefulness_avg", "rag_clinical_usefulness_avg", "no_rag_clinical_usefulness_avg"),
            ("safety_avg", "rag_safety_avg", "no_rag_safety_avg"),
        ]
        for label, rk, nk in rows:
            lines.append(f"| {label} | {_pretty(metrics.get(rk))} | {_pretty(metrics.get(nk))} |")

        lines.append("\n**Citation metrics (RAG):**\n")
        lines.append("| Метрика | Значение |")
        lines.append("|---|---|")
        for k in (
            "rag_citation_coverage_claim_level",
            "rag_citation_coverage_item_level",
            "rag_citation_validity_rate",
            "rag_citation_accuracy_rate",
        ):
            lines.append(f"| {k} | {_pretty(metrics.get(k))} |")

        lines.append("\n**Comparison:**\n")
        lines.append("| Метрика | Значение |")
        lines.append("|---|---|")
        for k in (
            "faithfulness_soft_improvement_abs",
            "faithfulness_soft_improvement_rel",
            "hallucination_rate_reduction_abs",
            "case_level_wins_rag_better",
            "case_level_losses_rag_worse",
            "case_level_ties",
        ):
            lines.append(f"| {k} | {_pretty(metrics.get(k))} |")

        lines.append("\n## Target thresholds (ВКР)\n")
        lines.append("| Цель | Pass/Fail |")
        lines.append("|---|---|")
        for label, mk in (
            ("RAG faithfulness_soft >= 0.85", "targets_met_faithfulness"),
            ("RAG hallucination_rate < 0.15", "targets_met_hallucination"),
            ("RAG citation_accuracy_rate >= 0.90", "targets_met_citation_accuracy"),
            ("Faithfulness improvement >= 0.20", "targets_met_faithfulness_improvement"),
            ("Hallucination reduction >= 0.20", "targets_met_hallucination_reduction"),
        ):
            v = metrics.get(mk)
            lines.append(f"| {label} | {_check_md(v)} |")

    return "\n".join(lines) + "\n"


def _check_md(v: Any) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    return "—"


def _pretty(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _format_summary_md(
    *,
    experiment_name: str,
    status: Dict[str, Any],
    cfg: Dict[str, Any],
    retrieval: Dict[str, Any],
    reranking: Dict[str, Any],
    generation: Dict[str, Any],
    files_produced: List[str],
    llm_eval: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"# Experiment summary: `{experiment_name}`\n")
    lines.append(f"- Status: **{status.get('status', 'unknown')}**")
    if cfg.get("experiment", {}).get("description"):
        lines.append(f"- Description: {cfg['experiment']['description']}")
    lines.append(f"- Started: {status.get('started_at', '—')}")
    lines.append(f"- Finished: {status.get('finished_at', '—')}")
    lines.append(f"- Total runtime, sec: {status.get('total_runtime_sec', '—')}\n")

    lines.append("## Experiment config\n")
    lines.append(f"- output_dir: `{cfg.get('experiment', {}).get('output_dir')}`")
    lines.append(f"- retrieval_queries_path: `{cfg.get('data', {}).get('retrieval_queries_path')}`")
    lines.append(f"- clinical_cases_path: `{cfg.get('data', {}).get('clinical_cases_path')}`")
    lines.append(f"- prompt_config_rag: `{cfg.get('generation', {}).get('prompt_config_rag')}`")
    lines.append(f"- prompt_config_no_rag: `{cfg.get('generation', {}).get('prompt_config_no_rag')}`\n")

    lines.append("## Retrieval results\n")
    if retrieval.get("status") == "skipped":
        lines.append("_Skipped._\n")
    else:
        lines.append(f"- best_embedding_model: `{retrieval.get('best_embedding_model')}`")
        lines.append(f"- best_recall_at_5: {_pretty(retrieval.get('best_recall_at_5'))}")
        lines.append(f"- best_precision_at_5: {_pretty(retrieval.get('best_precision_at_5'))}")
        lines.append(f"- best_mrr: {_pretty(retrieval.get('best_mrr'))}\n")

    lines.append("## Reranking / context-selection results\n")
    if reranking.get("status") == "skipped":
        lines.append("_Skipped._\n")
    else:
        for k in (
            "embedding_model",
            "reranker",
            "context_selection",
            "final_document_recall_at_5",
            "final_document_precision_at_5",
            "final_chunk_hit_at_5",
            "final_page_hit_at_5",
        ):
            lines.append(f"- {k}: {_pretty(reranking.get(k))}")
        lines.append("")

    lines.append("## Generation results\n")
    if generation.get("status") == "skipped":
        lines.append("_Skipped._\n")
    else:
        for k in (
            "llm_provider",
            "llm_model_name",
            "num_rag_answers",
            "num_no_rag_answers",
            "rag_valid_json_rate",
            "no_rag_valid_json_rate",
            "rag_avg_citation_count",
            "rag_avg_citation_coverage_estimate",
            "rag_invalid_citations_total",
        ):
            lines.append(f"- {k}: {_pretty(generation.get(k))}")
        lines.append("")

    if llm_eval and llm_eval.get("status") == "completed":
        lines.append("## LLM evaluation results\n")
        rag = llm_eval.get("rag") or {}
        no_rag = llm_eval.get("no_rag") or {}
        cmp_ = llm_eval.get("comparison") or {}
        lines.append("| Метрика | RAG | no-RAG |")
        lines.append("|---|---|---|")
        for label, rk, nk in (
            ("faithfulness_soft", "faithfulness_soft", "faithfulness_soft"),
            ("hallucination_rate", "hallucination_rate", "hallucination_rate"),
            ("answer_relevance_avg", "answer_relevance_avg", "answer_relevance_avg"),
            ("clinical_usefulness_avg", "clinical_usefulness_avg", "clinical_usefulness_avg"),
            ("safety_avg", "safety_avg", "safety_avg"),
        ):
            lines.append(f"| {label} | {_pretty(rag.get(rk))} | {_pretty(no_rag.get(nk))} |")
        lines.append("")
        lines.append("**Comparison:**\n")
        for k in (
            "faithfulness_soft_improvement_abs",
            "hallucination_rate_reduction_abs",
            "case_level_wins_rag_better",
            "case_level_losses_rag_worse",
        ):
            lines.append(f"- {k}: {_pretty(cmp_.get(k))}")
        lines.append("")
        lines.append("**Targets met:**\n")
        for label, mk in (
            ("RAG faithfulness_soft >= 0.85", "rag_meets_faithfulness_target"),
            ("RAG hallucination_rate < 0.15", "rag_meets_hallucination_target"),
            ("RAG citation_accuracy_rate >= 0.90", "rag_meets_citation_accuracy_target"),
            ("Faithfulness improvement >= 0.20", "rag_meets_faithfulness_improvement_target"),
            ("Hallucination reduction >= 0.20", "rag_meets_hallucination_reduction_target"),
        ):
            v = cmp_.get(mk)
            lines.append(f"- {label}: {_check_md(v)}")
        lines.append("")
        lines.append("Подробный отчёт: `llm_eval/llm_eval_report.md`\n")

    lines.append("## Files produced\n")
    for f in files_produced:
        lines.append(f"- `{f}`")
    lines.append("")

    lines.append("## Next steps\n")
    if not (llm_eval and llm_eval.get("status") == "completed"):
        lines.append("- Run LLM evaluator: `python -m src.evaluate_llm_answers --pairs-path generation/rag_vs_no_rag_pairs.jsonl ...`")
    lines.append("- Inspect detailed records: `rerank/detailed_results.jsonl` для retrieval-debug.")
    lines.append("- Если метрики просели — обновить prompt-config или поменять reranker / context_selection.")
    return "\n".join(lines) + "\n"



def collect_summary(
    output_dir: str | Path,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    pipeline_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    ensure_dir(out_dir / "summary")
    ensure_dir(out_dir / "metrics")

    retrieval = collect_retrieval(out_dir)
    reranking = collect_reranking(out_dir)
    generation = collect_generation(out_dir)
    comparison = collect_comparison(out_dir)
    llm_eval = collect_llm_eval(out_dir)

    final_metrics = build_final_metrics(
        out_dir,
        cfg=cfg,
        retrieval=retrieval,
        reranking=reranking,
        generation=generation,
        llm_eval=llm_eval,
    )

    pipeline_status = pipeline_status or _safe_read_json(out_dir / "metrics" / "pipeline_status.json") or {}

    final_metrics_with_status = dict(final_metrics)
    final_metrics_with_status["status"] = pipeline_status.get("status", "unknown")
    final_metrics_with_status["started_at"] = pipeline_status.get("started_at")
    final_metrics_with_status["finished_at"] = pipeline_status.get("finished_at")
    final_metrics_with_status["total_runtime_sec"] = pipeline_status.get("total_runtime_sec")

    write_json(out_dir / "metrics" / "final_metrics_summary.json", final_metrics_with_status)
    _write_csv_row(final_metrics_with_status, out_dir / "metrics" / "final_metrics_summary.csv")
    (out_dir / "metrics" / "final_metrics_summary.md").write_text(
        _format_md(final_metrics_with_status, status=pipeline_status), encoding="utf-8"
    )

    rag_m = generation.get("rag") or None
    no_rag_m = generation.get("no_rag") or None
    gen_rows: List[Dict[str, Any]] = [r for r in (rag_m, no_rag_m) if r]
    if gen_rows:
        write_json(out_dir / "metrics" / "generation_metrics.json", {"per_mode": gen_rows})
        _write_csv_rows(gen_rows, out_dir / "metrics" / "generation_metrics.csv")
    elif (cfg or {}).get("generation", {}).get("enabled") is False:
        write_json(out_dir / "metrics" / "generation_metrics.json",
                   {"enabled": False, "status": "skipped"})

    experiment_name = (cfg or {}).get("experiment", {}).get("name") or out_dir.name
    files_produced = _list_produced_files(out_dir)

    summary_json = {
        "experiment_name": experiment_name,
        "retrieval": retrieval,
        "reranking": reranking,
        "generation": generation,
        "comparison": comparison,
        "llm_evaluation": llm_eval,
        "final_metrics": final_metrics,
        "status": pipeline_status.get("status", "unknown"),
        "files_produced": files_produced,
    }
    write_json(out_dir / "summary" / "experiment_summary.json", summary_json)

    md = _format_summary_md(
        experiment_name=experiment_name,
        status=pipeline_status,
        cfg=cfg or {},
        retrieval=retrieval,
        reranking=reranking,
        generation=generation,
        files_produced=files_produced,
        llm_eval=llm_eval,
    )
    (out_dir / "summary" / "experiment_summary.md").write_text(md, encoding="utf-8")

    return summary_json


def _write_csv_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})


def _list_produced_files(out_dir: Path) -> List[str]:
    files: List[str] = []
    for sub in ("retrieval", "rerank", "generation", "metrics", "summary"):
        d = out_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                files.append(str(p.relative_to(out_dir)))
    return files


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect experiment summary")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--config", default=None, type=str,
                   help="Optional experiment config (для имени эксперимента и прочих путей).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cfg: Optional[Dict[str, Any]] = None
    if args.config:
        from .experiment_config import load_experiment_config
        cfg = load_experiment_config(args.config)
    summary = collect_summary(args.output_dir, cfg=cfg)
    print(json.dumps({k: summary.get(k) for k in ("experiment_name", "status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
