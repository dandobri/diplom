from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import read_yaml



def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    cfg = read_yaml(path)
    if isinstance(cfg.get("extends"), str):
        base_path = (Path(path).parent / cfg["extends"]).resolve()
        base = read_yaml(base_path)
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "extends"})
    return cfg


def validate_experiment_config(cfg: Dict[str, Any]) -> None:
    errors: List[str] = []

    exp = cfg.get("experiment") or {}
    if not exp.get("name"):
        errors.append("experiment.name is required")
    if not exp.get("output_dir"):
        errors.append("experiment.output_dir is required")

    data = cfg.get("data") or {}

    retr = cfg.get("retrieval_eval") or {}
    if retr.get("enabled"):
        if not data.get("retrieval_queries_path"):
            errors.append("data.retrieval_queries_path is required when retrieval_eval.enabled=true")
        if not retr.get("config_path"):
            errors.append("retrieval_eval.config_path is required when retrieval_eval.enabled=true")
        if not cfg.get("artifacts", {}).get("embeddings_root"):
            errors.append("artifacts.embeddings_root is required when retrieval_eval.enabled=true")

    rer = cfg.get("reranking_eval") or {}
    if rer.get("enabled"):
        if not data.get("retrieval_queries_path"):
            errors.append("data.retrieval_queries_path is required when reranking_eval.enabled=true")
        if not rer.get("embeddings_dir"):
            errors.append("reranking_eval.embeddings_dir is required when reranking_eval.enabled=true")
        if not rer.get("embedding_model_key"):
            errors.append("reranking_eval.embedding_model_key is required when reranking_eval.enabled=true")
        if not rer.get("embedding_config"):
            errors.append("reranking_eval.embedding_config is required when reranking_eval.enabled=true")
        if not rer.get("reranker_key"):
            errors.append("reranking_eval.reranker_key is required when reranking_eval.enabled=true")
        if not rer.get("reranker_config"):
            errors.append("reranking_eval.reranker_config is required when reranking_eval.enabled=true")

    gen = cfg.get("generation") or {}
    if gen.get("enabled"):
        if not data.get("clinical_cases_path"):
            errors.append("data.clinical_cases_path is required when generation.enabled=true")
        if not gen.get("llm_config"):
            errors.append("generation.llm_config is required when generation.enabled=true")
        if gen.get("run_rag", True) and not gen.get("prompt_config_rag"):
            errors.append("generation.prompt_config_rag is required when generation.run_rag=true")
        if gen.get("run_no_rag", True) and not gen.get("prompt_config_no_rag"):
            errors.append("generation.prompt_config_no_rag is required when generation.run_no_rag=true")
        if gen.get("run_rag", True):
            if not rer.get("embeddings_dir") and not gen.get("embeddings_dir"):
                errors.append(
                    "generation.run_rag=true requires retrieval artifacts: укажите "
                    "reranking_eval.embeddings_dir или generation.embeddings_dir"
                )

    cmp_ = cfg.get("comparison") or {}
    if cmp_.get("enabled"):
        if not gen.get("enabled"):
            errors.append("comparison.enabled=true requires generation.enabled=true")
        if gen.get("enabled") and not (gen.get("run_rag", True) and gen.get("run_no_rag", True)):
            errors.append("comparison.enabled=true requires both run_rag and run_no_rag")

    llm_eval = cfg.get("llm_evaluation") or {}
    if llm_eval.get("enabled"):
        if not llm_eval.get("llm_config"):
            errors.append("llm_evaluation.llm_config is required when llm_evaluation.enabled=true")
        if not llm_eval.get("judge_prompt_config"):
            errors.append("llm_evaluation.judge_prompt_config is required when llm_evaluation.enabled=true")
        m = llm_eval.get("mode") or "both"
        if m not in ("both", "rag", "no_rag"):
            errors.append(f"llm_evaluation.mode must be one of both/rag/no_rag, got {m!r}")
        pp = llm_eval.get("pairs_path")
        rp = llm_eval.get("rag_answers_path")
        np_ = llm_eval.get("no_rag_answers_path")
        if pp and (rp or np_):
            errors.append(
                "llm_evaluation: укажите либо pairs_path, либо "
                "rag_answers_path/no_rag_answers_path, не оба."
            )

    if errors:
        joined = "\n  - " + "\n  - ".join(errors)
        raise ValueError(f"Invalid experiment config:{joined}")


def _g(cfg: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return (cfg.get(section) or {}).get(key, default)


def resolve_experiment_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(_g(cfg, "experiment", "output_dir"))

    paths: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "config_resolved": str(out_dir / "config_resolved.yaml"),
        "log_file": str(out_dir / "experiment.log"),
        "datasphere_commands": str(out_dir / "datasphere_pipeline_commands.md"),
        "retrieval": {
            "dir": str(out_dir / "retrieval"),
            "comparison_csv": str(out_dir / "retrieval" / "reports" / "embedding_model_comparison.csv"),
            "comparison_json": str(out_dir / "retrieval" / "reports" / "embedding_model_comparison.json"),
            "best_model_json": str(out_dir / "retrieval" / "reports" / "best_embedding_model.json"),
            "results_dir": str(out_dir / "retrieval" / "retrieval_results"),
        },
        "rerank": {
            "dir": str(out_dir / "rerank"),
            "detailed": str(out_dir / "rerank" / "detailed_results.jsonl"),
            "comparison_csv": str(out_dir / "rerank" / "rerank_comparison_metrics.csv"),
            "comparison_json": str(out_dir / "rerank" / "rerank_comparison_metrics.json"),
            "report_md": str(out_dir / "rerank" / "rerank_report.md"),
        },
        "generation": {
            "dir": str(out_dir / "generation"),
            "rag_answers": str(out_dir / "generation" / "rag_answers.jsonl"),
            "no_rag_answers": str(out_dir / "generation" / "no_rag_answers.jsonl"),
            "pairs": str(out_dir / "generation" / "rag_vs_no_rag_pairs.jsonl"),
            "summary_csv": str(out_dir / "generation" / "rag_vs_no_rag_summary.csv"),
        },
        "llm_eval": {
            "dir": str(out_dir / "llm_eval"),
            "claim_evaluations": str(out_dir / "llm_eval" / "claim_evaluations.jsonl"),
            "case_evaluations": str(out_dir / "llm_eval" / "case_evaluations.jsonl"),
            "summary_csv": str(out_dir / "llm_eval" / "rag_vs_no_rag_eval_summary.csv"),
            "summary_json": str(out_dir / "llm_eval" / "rag_vs_no_rag_eval_summary.json"),
            "metrics_json": str(out_dir / "llm_eval" / "llm_eval_metrics.json"),
            "metrics_csv": str(out_dir / "llm_eval" / "llm_eval_metrics.csv"),
            "report_md": str(out_dir / "llm_eval" / "llm_eval_report.md"),
            "failed": str(out_dir / "llm_eval" / "failed_judge_cases.jsonl"),
            "log_file": str(out_dir / "llm_eval" / "evaluate_llm_answers.log"),
        },
        "metrics": {
            "dir": str(out_dir / "metrics"),
            "pipeline_status": str(out_dir / "metrics" / "pipeline_status.json"),
            "retrieval_json": str(out_dir / "metrics" / "retrieval_metrics.json"),
            "retrieval_csv": str(out_dir / "metrics" / "retrieval_metrics.csv"),
            "rerank_json": str(out_dir / "metrics" / "rerank_metrics.json"),
            "rerank_csv": str(out_dir / "metrics" / "rerank_metrics.csv"),
            "generation_json": str(out_dir / "metrics" / "generation_metrics.json"),
            "generation_csv": str(out_dir / "metrics" / "generation_metrics.csv"),
            "llm_eval_json": str(out_dir / "metrics" / "llm_eval_metrics.json"),
            "llm_eval_csv": str(out_dir / "metrics" / "llm_eval_metrics.csv"),
            "final_summary_json": str(out_dir / "metrics" / "final_metrics_summary.json"),
            "final_summary_csv": str(out_dir / "metrics" / "final_metrics_summary.csv"),
            "final_summary_md": str(out_dir / "metrics" / "final_metrics_summary.md"),
        },
        "summary": {
            "dir": str(out_dir / "summary"),
            "json": str(out_dir / "summary" / "experiment_summary.json"),
            "md": str(out_dir / "summary" / "experiment_summary.md"),
        },
    }
    return paths


def assert_overwrite_allowed(cfg: Dict[str, Any]) -> None:
    out_dir = Path(_g(cfg, "experiment", "output_dir"))
    overwrite = bool(_g(cfg, "experiment", "overwrite", False))
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Experiment output_dir already exists and is not empty: {out_dir}. "
            "Set experiment.overwrite=true or pass --overwrite to allow."
        )


def apply_cli_overrides(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return cfg
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("experiment", {})
    cfg.setdefault("retrieval_eval", {})
    cfg.setdefault("reranking_eval", {})
    cfg.setdefault("generation", {})
    cfg.setdefault("comparison", {})
    cfg.setdefault("llm_evaluation", {})

    if overrides.get("output_dir") is not None:
        cfg["experiment"]["output_dir"] = overrides["output_dir"]
    if overrides.get("overwrite") is not None:
        cfg["experiment"]["overwrite"] = bool(overrides["overwrite"])
    if overrides.get("limit") is not None:
        cfg["generation"]["limit"] = int(overrides["limit"])
        cfg["llm_evaluation"]["limit"] = int(overrides["limit"])
    if overrides.get("device") is not None:
        cfg["reranking_eval"]["device"] = overrides["device"]
        cfg["generation"]["device"] = overrides["device"]
    if overrides.get("skip_retrieval"):
        cfg["retrieval_eval"]["enabled"] = False
    if overrides.get("skip_rerank"):
        cfg["reranking_eval"]["enabled"] = False
    if overrides.get("skip_generation"):
        cfg["generation"]["enabled"] = False
    if overrides.get("skip_comparison"):
        cfg["comparison"]["enabled"] = False
    if overrides.get("skip_llm_eval"):
        cfg["llm_evaluation"]["enabled"] = False
    return cfg
