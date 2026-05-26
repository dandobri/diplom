from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compare_rag_vs_no_rag import merge as compare_merge
from .evaluate_llm_answers import run_llm_evaluation
from .experiment_config import load_experiment_config
from .generate_answers import run_generation
from .io_utils import ensure_dir, read_json, setup_logging, write_json
from .prompt_templates import load_prompt_config

logger = logging.getLogger(__name__)


def _prompt_short_name(path: str) -> str:
    
    try:
        cfg = load_prompt_config(path)
        return cfg.get("name") or Path(path).stem
    except Exception:  
        return Path(path).stem


def _run_one_prompt(
    *,
    prompt_path: str,
    prompt_name: str,
    cfg: Dict[str, Any],
    output_dir: Path,
    limit: Optional[int],
    no_rag_answers_path: Optional[Path],
) -> Dict[str, Any]:
    
    out = output_dir / prompt_name
    ensure_dir(out / "generation")
    ensure_dir(out / "llm_eval")

    gen_section = cfg["generation"]
    rer_section = cfg.get("reranking_eval") or {}

    rag_answers = out / "generation" / "rag_answers.jsonl"
    no_rag_answers = out / "generation" / "no_rag_answers.jsonl"

    
    logger.info("[%s] step 1/3: RAG generation", prompt_name)
    run_generation(
        mode="rag",
        cases_path=cfg["data"]["clinical_cases_path"],
        output_path=rag_answers,
        llm_config=gen_section["llm_config"],
        llm_provider=gen_section.get("llm_provider"),
        llm_model=gen_section.get("llm_model"),
        prompt_config_path=prompt_path,
        case_id_field=(gen_section.get("mode_cases_field") or {}).get("case_id_field") or "case_id",
        patient_case_field=(gen_section.get("mode_cases_field") or {}).get("patient_case_field") or "patient_case",
        limit=limit,
        overwrite=True,
        resume=False,
        embedding_model_key=gen_section.get("embedding_model_key") or rer_section.get("embedding_model_key"),
        embedding_config=gen_section.get("embedding_config") or rer_section.get("embedding_config"),
        embeddings_dir=gen_section.get("embeddings_dir") or rer_section.get("embeddings_dir"),
        reranker_key=gen_section.get("reranker_key") or rer_section.get("reranker_key"),
        reranker_config=gen_section.get("reranker_config") or rer_section.get("reranker_config"),
        candidate_top_k=gen_section.get("candidate_top_k") or rer_section.get("candidate_top_k") or 30,
        final_top_k=gen_section.get("final_top_k") or rer_section.get("final_top_k") or 5,
        context_selection=gen_section.get("context_selection") or rer_section.get("context_selection") or "anchor_page",
        context_page_tolerance=gen_section.get("context_page_tolerance") or rer_section.get("context_page_tolerance") or 1,
        device=gen_section.get("device") or "auto",
    )

    
    if no_rag_answers_path and no_rag_answers_path.exists():
        logger.info("[%s] step 2/3: reuse shared no-RAG → copy", prompt_name)
        shutil.copy(no_rag_answers_path, no_rag_answers)
    else:
        logger.info("[%s] step 2/3: no-RAG generation", prompt_name)
        run_generation(
            mode="no_rag",
            cases_path=cfg["data"]["clinical_cases_path"],
            output_path=no_rag_answers,
            llm_config=gen_section["llm_config"],
            prompt_config_path=gen_section["prompt_config_no_rag"],
            case_id_field=(gen_section.get("mode_cases_field") or {}).get("case_id_field") or "case_id",
            patient_case_field=(gen_section.get("mode_cases_field") or {}).get("patient_case_field") or "patient_case",
            limit=limit,
            overwrite=True,
            resume=False,
        )

    
    pairs_path = out / "generation" / "rag_vs_no_rag_pairs.jsonl"
    summary_csv = out / "generation" / "rag_vs_no_rag_summary.csv"
    compare_merge(
        rag_path=str(rag_answers),
        no_rag_path=str(no_rag_answers),
        output_pairs=str(pairs_path),
        output_summary=str(summary_csv),
    )

    
    logger.info("[%s] step 3/3: LLM evaluation", prompt_name)
    le = cfg["llm_evaluation"]
    info = run_llm_evaluation(
        pairs_path=str(pairs_path),
        rag_answers_path=None,
        no_rag_answers_path=None,
        clinical_cases_path=cfg["data"]["clinical_cases_path"],
        llm_config=le["llm_config"],
        judge_prompt_config=le["judge_prompt_config"],
        citation_judge_prompt_config=le.get("citation_judge_prompt_config"),
        relevance_judge_prompt_config=le.get("relevance_judge_prompt_config"),
        output_dir=out / "llm_eval",
        limit=limit,
        overwrite=True,
        resume=False,
        mode="both",
        judge_provider=le.get("judge_provider"),
        judge_model=le.get("judge_model"),
        max_claims_per_answer=int(le.get("max_claims_per_answer") or 20),
        judge_max_retries=int(le.get("judge_max_retries") or 2),
    )
    metrics = read_json(info["metrics_json"])

    return {
        "prompt_name": prompt_name,
        "prompt_path": prompt_path,
        "output_dir": str(out),
        "metrics": metrics,
    }


def _flatten_for_sweep(prompt_name: str, prompt_path: str, m: Dict[str, Any]) -> Dict[str, Any]:
    rag = m.get("rag") or {}
    no_rag = m.get("no_rag") or {}
    cmp_ = m.get("comparison") or {}
    return {
        "prompt_name": prompt_name,
        "prompt_path": prompt_path,
        "num_cases": m.get("num_cases"),
        "rag_judged_cases": rag.get("judged_cases"),
        "rag_judge_success_rate": rag.get("judge_success_rate"),
        "no_rag_judged_cases": no_rag.get("judged_cases"),
        "no_rag_judge_success_rate": no_rag.get("judge_success_rate"),
        "rag_total_claims": rag.get("total_claims"),
        "no_rag_total_claims": no_rag.get("total_claims"),
        "rag_avg_claims_per_answer": (
            round(rag.get("total_claims") / rag.get("num_cases"), 2)
            if rag.get("num_cases") else None
        ),
        "rag_faithfulness_strict": rag.get("faithfulness_strict"),
        "rag_faithfulness_soft": rag.get("faithfulness_soft"),
        "no_rag_faithfulness_soft": no_rag.get("faithfulness_soft"),
        "rag_hallucination_rate": rag.get("hallucination_rate"),
        "no_rag_hallucination_rate": no_rag.get("hallucination_rate"),
        "rag_citation_coverage_claim_level": rag.get("citation_coverage_claim_level"),
        "rag_citation_validity_rate": rag.get("citation_validity_rate"),
        "rag_citation_accuracy_rate": rag.get("citation_accuracy_rate"),
        "rag_answer_relevance_avg": rag.get("answer_relevance_avg"),
        "rag_clinical_usefulness_avg": rag.get("clinical_usefulness_avg"),
        "rag_safety_avg": rag.get("safety_avg"),
        "faithfulness_soft_improvement_abs": cmp_.get("faithfulness_soft_improvement_abs"),
        "hallucination_rate_reduction_abs": cmp_.get("hallucination_rate_reduction_abs"),
        "rag_meets_faithfulness": cmp_.get("rag_meets_faithfulness_target"),
        "rag_meets_hallucination": cmp_.get("rag_meets_hallucination_target"),
        "rag_meets_citation_accuracy": cmp_.get("rag_meets_citation_accuracy_target"),
        "rag_meets_faithfulness_improvement": cmp_.get("rag_meets_faithfulness_improvement_target"),
        "rag_meets_hallucination_reduction": cmp_.get("rag_meets_hallucination_reduction_target"),
    }


def _write_sweep_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    csv_path = out_dir / "sweep_summary.csv"
    md_path = out_dir / "sweep_summary.md"

    cols = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})

    
    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "✓" if v else "✗"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines: List[str] = []
    lines.append("# Prompt sweep — comparison\n")
    lines.append("Все варианты прогнаны на одном и том же:\n")
    lines.append("- наборе clinical cases;\n")
    lines.append("- одном и том же no-RAG ответе (общий baseline);\n")
    lines.append("- одних и тех же judge promptах.\n")
    lines.append("\nЕдинственное различие между строками — RAG prompt.\n")

    main_cols = [
        "prompt_name",
        "rag_judge_success_rate",
        "rag_avg_claims_per_answer",
        "rag_faithfulness_soft",
        "rag_hallucination_rate",
        "rag_citation_accuracy_rate",
        "faithfulness_soft_improvement_abs",
        "hallucination_rate_reduction_abs",
    ]
    lines.append("\n## Главное\n")
    lines.append("| " + " | ".join(main_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(main_cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in main_cols) + " |")

    target_cols = [
        "prompt_name",
        "rag_meets_faithfulness",
        "rag_meets_hallucination",
        "rag_meets_citation_accuracy",
        "rag_meets_faithfulness_improvement",
        "rag_meets_hallucination_reduction",
    ]
    lines.append("\n## Targets met\n")
    lines.append("| " + " | ".join(target_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(target_cols)) + "|")
    for r in rows:
        passes = sum(
            1 for c in target_cols[1:] if r.get(c) is True
        )
        row_cells = [_fmt(r.get(c)) for c in target_cols]
        lines.append("| " + " | ".join(row_cells) + f" | ({passes}/5) |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote sweep summary → %s", md_path)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt sweep — controlled prompt comparison")
    p.add_argument("--config", required=True, type=str,
                   help="Experiment YAML (используется для retrieval/llm/cases path).")
    p.add_argument("--prompts", required=True, nargs="+",
                   help="Список путей к RAG prompt YAML — по одному прогону на каждый.")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--limit", type=int, default=20,
                   help="Сколько кейсов прогнать на каждом prompt (default 20).")
    p.add_argument("--no-rag-shared", action="store_true", default=True,
                   help="Сгенерировать no-RAG один раз и переиспользовать (default true).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    setup_logging(verbose=args.verbose, log_file=out_dir / "prompt_sweep.log", name="sweep")

    cfg = load_experiment_config(args.config)
    logger.info("Loaded config: %s", args.config)

    
    no_rag_shared_path: Optional[Path] = None
    if args.no_rag_shared:
        shared = out_dir / "_shared"
        ensure_dir(shared / "generation")
        no_rag_shared_path = shared / "generation" / "no_rag_answers.jsonl"
        if not no_rag_shared_path.exists():
            logger.info("Generating shared no-RAG baseline (one-shot)...")
            gen = cfg["generation"]
            run_generation(
                mode="no_rag",
                cases_path=cfg["data"]["clinical_cases_path"],
                output_path=no_rag_shared_path,
                llm_config=gen["llm_config"],
                prompt_config_path=gen["prompt_config_no_rag"],
                case_id_field=(gen.get("mode_cases_field") or {}).get("case_id_field") or "case_id",
                patient_case_field=(gen.get("mode_cases_field") or {}).get("patient_case_field") or "patient_case",
                limit=args.limit,
                overwrite=True,
                resume=False,
            )
        else:
            logger.info("Shared no-RAG already exists at %s — reusing", no_rag_shared_path)

    rows: List[Dict[str, Any]] = []
    for prompt_path in args.prompts:
        prompt_name = _prompt_short_name(prompt_path)
        logger.info("=== Sweep prompt: %s (%s) ===", prompt_name, prompt_path)
        try:
            res = _run_one_prompt(
                prompt_path=prompt_path,
                prompt_name=prompt_name,
                cfg=cfg,
                output_dir=out_dir,
                limit=args.limit,
                no_rag_answers_path=no_rag_shared_path,
            )
            row = _flatten_for_sweep(prompt_name, prompt_path, res["metrics"])
            rows.append(row)
        except Exception as e:  
            logger.exception("Sweep failed for prompt=%s", prompt_name)
            rows.append({
                "prompt_name": prompt_name,
                "prompt_path": prompt_path,
                "error": str(e),
            })

    write_json(out_dir / "sweep_results.json", {"rows": rows})
    if rows and "error" not in rows[0]:
        _write_sweep_summary(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
