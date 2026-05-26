from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from .compare_rag_vs_no_rag import merge as compare_merge
from .collect_experiment_summary import (
    _generation_metrics_for,
    collect_generation,
    collect_reranking,
    collect_retrieval,
    collect_summary,
)
from .evaluate_llm_answers import run_llm_evaluation
from .evaluate_models import evaluate_models as run_retrieval_eval
from .evaluate_with_reranker import run_evaluation as run_rerank_eval
from .experiment_config import (
    apply_cli_overrides,
    assert_overwrite_allowed,
    load_experiment_config,
    resolve_experiment_paths,
    validate_experiment_config,
)
from .generate_answers import run_generation
from .io_utils import ensure_dir, read_json, read_jsonl_list, setup_logging, write_json
from .write_datasphere_commands import write_commands as write_datasphere_commands

logger = logging.getLogger(__name__)




def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineStatus:
    """Тонкий wrapper над pipeline_status.json — пишет на диск после каждого
    обновления, чтобы прогресс был видно даже при crash'е."""

    def __init__(self, *, experiment_name: str, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "total_runtime_sec": None,
            "steps": {},
        }
        ensure_dir(self.path.parent)
        self._t0 = time.perf_counter()
        self.flush()

    def step_start(self, name: str, *, enabled: bool = True) -> None:
        self.data["steps"][name] = {
            "enabled": enabled,
            "status": "running" if enabled else "skipped",
            "started_at": _utc_now() if enabled else None,
            "finished_at": None,
        }
        self.flush()

    def step_skip(self, name: str) -> None:
        self.data["steps"][name] = {
            "enabled": False,
            "status": "skipped",
        }
        self.flush()

    def step_complete(
        self,
        name: str,
        *,
        key_metrics: Optional[Dict[str, Any]] = None,
        output_files: Optional[List[str]] = None,
    ) -> None:
        s = self.data["steps"].setdefault(name, {})
        s["status"] = "completed"
        s["finished_at"] = _utc_now()
        if key_metrics:
            s["key_metrics"] = key_metrics
        if output_files:
            s["output_files"] = output_files
        self.flush()

    def step_fail(self, name: str, *, error: BaseException, output_files: Optional[List[str]] = None) -> None:
        s = self.data["steps"].setdefault(name, {})
        s["status"] = "failed"
        s["finished_at"] = _utc_now()
        s["error_message"] = str(error)
        s["traceback"] = traceback.format_exc()
        if output_files:
            s["output_files"] = output_files
        self.flush()

    def finalize(self) -> None:
        steps = self.data.get("steps", {})
        any_failed = any(v.get("status") == "failed" for v in steps.values())
        any_completed = any(v.get("status") == "completed" for v in steps.values())
        if any_failed and any_completed:
            self.data["status"] = "completed_with_errors"
        elif any_failed:
            self.data["status"] = "failed"
        elif any_completed:
            self.data["status"] = "completed"
        else:
            self.data["status"] = "no_steps_run"
        self.data["finished_at"] = _utc_now()
        self.data["total_runtime_sec"] = round(time.perf_counter() - self._t0, 3)
        self.flush()

    def flush(self) -> None:
        write_json(self.path, self.data)




def _write_step_metrics(
    *,
    metrics_dir: Path,
    name_prefix: str,
    json_payload: Dict[str, Any],
    csv_payload: Optional[Dict[str, Any]] = None,
) -> List[str]:
    out_files: List[str] = []
    json_path = metrics_dir / f"{name_prefix}_metrics.json"
    write_json(json_path, json_payload)
    out_files.append(str(json_path))
    if csv_payload is not None:
        csv_path = metrics_dir / f"{name_prefix}_metrics.csv"
        cols = list(csv_payload.keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerow({k: ("" if csv_payload.get(k) is None else csv_payload[k]) for k in cols})
        out_files.append(str(csv_path))
    return out_files


def _write_skipped_metrics(metrics_dir: Path, name_prefix: str) -> None:
    write_json(
        metrics_dir / f"{name_prefix}_metrics.json",
        {"enabled": False, "status": "skipped"},
    )




def step_retrieval(
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    status: PipelineStatus,
) -> None:
    section = cfg.get("retrieval_eval") or {}
    name = "retrieval_eval"
    metrics_dir = Path(paths["metrics"]["dir"])

    if not section.get("enabled"):
        status.step_skip(name)
        _write_skipped_metrics(metrics_dir, "retrieval")
        return

    status.step_start(name, enabled=True)
    out_dir = Path(paths["retrieval"]["dir"])
    ensure_dir(out_dir)
    try:
        result = run_retrieval_eval(
            queries_path=cfg["data"]["retrieval_queries_path"],
            config_path=section["config_path"],
            embeddings_root=cfg["artifacts"]["embeddings_root"],
            model_keys=section.get("model_keys"),
            output_dir=str(out_dir),
            device=section.get("device", "auto"),
            top_k=section.get("top_k", 10),
        )
        best = (result or {}).get("best") or {}
        rows = (result or {}).get("rows") or []
        key_metrics: Dict[str, Any] = {
            "best_model": best.get("best_model_key"),
            "num_models": len(rows),
        }
        bm = best.get("metrics") or {}
        for k in ("recall_at_5", "precision_at_5", "mrr", "hit_at_5"):
            if k in bm:
                key_metrics[k] = bm[k]

        files = [
            paths["retrieval"]["comparison_csv"],
            paths["retrieval"]["comparison_json"],
            paths["retrieval"]["best_model_json"],
        ]
        
        csv_row: Dict[str, Any] = {
            "best_model_key": best.get("best_model_key"),
            "reason": best.get("reason"),
        }
        csv_row.update(bm)
        _write_step_metrics(
            metrics_dir=metrics_dir,
            name_prefix="retrieval",
            json_payload={
                "enabled": True,
                "status": "completed",
                "best": best,
                "rows": rows,
            },
            csv_payload=csv_row,
        )
        status.step_complete(name, key_metrics=key_metrics, output_files=files)
    except Exception as e:  
        logger.exception("retrieval_eval failed")
        _write_step_metrics(
            metrics_dir=metrics_dir,
            name_prefix="retrieval",
            json_payload={"enabled": True, "status": "failed", "error": str(e)},
        )
        status.step_fail(name, error=e)


def step_rerank(
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    status: PipelineStatus,
) -> None:
    section = cfg.get("reranking_eval") or {}
    name = "reranking_eval"
    metrics_dir = Path(paths["metrics"]["dir"])

    if not section.get("enabled"):
        status.step_skip(name)
        _write_skipped_metrics(metrics_dir, "rerank")
        return

    status.step_start(name, enabled=True)
    out_dir = Path(paths["rerank"]["dir"])
    ensure_dir(out_dir)
    try:
        result = run_rerank_eval(
            queries_path=cfg["data"]["retrieval_queries_path"],
            embedding_model_key=section["embedding_model_key"],
            embedding_config=section["embedding_config"],
            embeddings_dir=section["embeddings_dir"],
            reranker_key=section["reranker_key"],
            reranker_config=section["reranker_config"],
            candidate_top_k=section.get("candidate_top_k", 30),
            final_top_k=section.get("final_top_k", 5),
            device=section.get("device", "auto"),
            output_dir=str(out_dir),
            limit=section.get("limit"),
            overwrite=True,
            save_all_candidates=section.get("save_all_candidates", False),
            context_selection=section.get("context_selection", "none"),
            context_page_tolerance=section.get("context_page_tolerance", 1),
        )
        final_m = (result or {}).get("final_metrics") or {}
        key_metrics = {
            "context_selection": result.get("context_selection"),
            "final_document_recall_at_5": final_m.get("document_recall_at_5"),
            "final_document_precision_at_5": final_m.get("document_precision_at_5"),
            "final_chunk_hit_at_5": final_m.get("chunk_hit_at_5"),
            "final_section_hit_at_5": final_m.get("section_hit_at_5"),
            "final_page_hit_at_5": final_m.get("page_hit_at_5"),
        }
        files = [
            paths["rerank"]["detailed"],
            paths["rerank"]["comparison_csv"],
            paths["rerank"]["comparison_json"],
            paths["rerank"]["report_md"],
        ]
        
        csv_row = {
            "embedding_model_key": section["embedding_model_key"],
            "reranker_key": section["reranker_key"],
            "context_selection": result.get("context_selection"),
            "num_queries": result.get("num_queries"),
            **{f"final_{k}": v for k, v in final_m.items()},
        }
        _write_step_metrics(
            metrics_dir=metrics_dir,
            name_prefix="rerank",
            json_payload={"enabled": True, "status": "completed", **(result or {})},
            csv_payload=csv_row,
        )
        status.step_complete(name, key_metrics=key_metrics, output_files=files)
    except Exception as e:  
        logger.exception("reranking_eval failed")
        _write_step_metrics(
            metrics_dir=metrics_dir,
            name_prefix="rerank",
            json_payload={"enabled": True, "status": "failed", "error": str(e)},
        )
        status.step_fail(name, error=e)


def _resolve_rag_retrieval_args(cfg: Dict[str, Any]) -> Dict[str, Any]:
    
    gen = cfg.get("generation") or {}
    rer = cfg.get("reranking_eval") or {}
    pick = lambda k: gen.get(k) if gen.get(k) is not None else rer.get(k)
    return {
        "embedding_model_key": pick("embedding_model_key"),
        "embedding_config": pick("embedding_config"),
        "embeddings_dir": pick("embeddings_dir"),
        "reranker_key": pick("reranker_key"),
        "reranker_config": pick("reranker_config"),
        "candidate_top_k": pick("candidate_top_k") or 30,
        "final_top_k": pick("final_top_k") or 5,
        "context_selection": pick("context_selection") or "anchor_page",
        "context_page_tolerance": pick("context_page_tolerance") or 1,
        "device": pick("device") or "auto",
    }


def step_generation(
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    status: PipelineStatus,
) -> None:
    section = cfg.get("generation") or {}
    metrics_dir = Path(paths["metrics"]["dir"])
    if not section.get("enabled"):
        status.step_skip("rag_generation")
        status.step_skip("no_rag_generation")
        _write_skipped_metrics(metrics_dir, "generation")
        return

    cases_path = cfg["data"]["clinical_cases_path"]
    case_field = (section.get("mode_cases_field") or {}).get("case_id_field")
    patient_field = (section.get("mode_cases_field") or {}).get("patient_case_field")
    limit = section.get("limit")
    rag_paths = _resolve_rag_retrieval_args(cfg)

    rag_metrics: Optional[Dict[str, Any]] = None
    no_rag_metrics: Optional[Dict[str, Any]] = None

    
    if section.get("run_rag", True):
        status.step_start("rag_generation", enabled=True)
        rag_out = Path(paths["generation"]["rag_answers"])
        ensure_dir(rag_out.parent)
        try:
            result = run_generation(
                mode="rag",
                cases_path=cases_path,
                output_path=rag_out,
                llm_config=section["llm_config"],
                llm_provider=section.get("llm_provider"),
                llm_model=section.get("llm_model"),
                prompt_config_path=section.get("prompt_config_rag"),
                case_id_field=case_field,
                patient_case_field=patient_field,
                limit=limit,
                overwrite=True,
                resume=False,
                **rag_paths,
            )
            rag_records = read_jsonl_list(rag_out) if rag_out.exists() else []
            rag_metrics = _generation_metrics_for(rag_records, "rag")
            key_metrics = {
                "num_answers": rag_metrics.get("num_cases"),
                "valid_json_rate": rag_metrics.get("valid_json_rate"),
                "avg_citation_count": rag_metrics.get("avg_citation_count"),
                "avg_citation_coverage": rag_metrics.get("avg_citation_coverage_estimate"),
                "invalid_citations_total": rag_metrics.get("invalid_citations_total"),
            }
            status.step_complete(
                "rag_generation",
                key_metrics=key_metrics,
                output_files=[str(rag_out)],
            )
        except Exception as e:  
            logger.exception("rag_generation failed")
            status.step_fail("rag_generation", error=e)
    else:
        status.step_skip("rag_generation")

    
    if section.get("run_no_rag", True):
        status.step_start("no_rag_generation", enabled=True)
        no_rag_out = Path(paths["generation"]["no_rag_answers"])
        ensure_dir(no_rag_out.parent)
        try:
            result = run_generation(
                mode="no_rag",
                cases_path=cases_path,
                output_path=no_rag_out,
                llm_config=section["llm_config"],
                llm_provider=section.get("llm_provider"),
                llm_model=section.get("llm_model"),
                prompt_config_path=section.get("prompt_config_no_rag"),
                case_id_field=case_field,
                patient_case_field=patient_field,
                limit=limit,
                overwrite=True,
                resume=False,
            )
            no_rag_records = read_jsonl_list(no_rag_out) if no_rag_out.exists() else []
            no_rag_metrics = _generation_metrics_for(no_rag_records, "no_rag")
            key_metrics = {
                "num_answers": no_rag_metrics.get("num_cases"),
                "valid_json_rate": no_rag_metrics.get("valid_json_rate"),
            }
            status.step_complete(
                "no_rag_generation",
                key_metrics=key_metrics,
                output_files=[str(no_rag_out)],
            )
        except Exception as e:  
            logger.exception("no_rag_generation failed")
            status.step_fail("no_rag_generation", error=e)
    else:
        status.step_skip("no_rag_generation")

    
    rows: List[Dict[str, Any]] = [r for r in (rag_metrics, no_rag_metrics) if r]
    if rows:
        write_json(
            metrics_dir / "generation_metrics.json",
            {"enabled": True, "status": "completed", "per_mode": rows},
        )
        cols: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        with (metrics_dir / "generation_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})


def step_comparison(
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    status: PipelineStatus,
) -> None:
    section = cfg.get("comparison") or {}
    name = "rag_vs_no_rag_comparison"
    if not section.get("enabled"):
        status.step_skip(name)
        return

    status.step_start(name, enabled=True)
    rag_path = Path(paths["generation"]["rag_answers"])
    no_rag_path = Path(paths["generation"]["no_rag_answers"])
    pairs = paths["generation"]["pairs"]
    summary_csv = paths["generation"]["summary_csv"]

    if not rag_path.exists() or not no_rag_path.exists():
        msg = (
            f"comparison skipped: rag_path exists={rag_path.exists()} "
            f"no_rag_path exists={no_rag_path.exists()}"
        )
        logger.warning(msg)
        status.step_fail(name, error=RuntimeError(msg))
        return

    try:
        info = compare_merge(
            rag_path=str(rag_path),
            no_rag_path=str(no_rag_path),
            output_pairs=pairs,
            output_summary=summary_csv,
        )
        status.step_complete(
            name,
            key_metrics={"num_cases": info.get("num_cases"), "both": info.get("both")},
            output_files=[pairs, summary_csv],
        )
    except Exception as e:  
        logger.exception("comparison failed")
        status.step_fail(name, error=e)


def step_llm_evaluation(
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    status: PipelineStatus,
) -> None:
    
    section = cfg.get("llm_evaluation") or {}
    name = "llm_evaluation"
    metrics_dir = Path(paths["metrics"]["dir"])

    if not section.get("enabled"):
        status.step_skip(name)
        return

    status.step_start(name, enabled=True)
    out_dir = Path(paths["llm_eval"]["dir"])
    ensure_dir(out_dir)

    
    pairs_path = section.get("pairs_path") or paths["generation"]["pairs"]
    rag_path = section.get("rag_answers_path")
    no_rag_path = section.get("no_rag_answers_path")

    try:
        info = run_llm_evaluation(
            pairs_path=pairs_path if pairs_path else None,
            rag_answers_path=rag_path,
            no_rag_answers_path=no_rag_path,
            clinical_cases_path=cfg.get("data", {}).get("clinical_cases_path"),
            llm_config=section.get("llm_config") or "configs/llm.yaml",
            judge_prompt_config=section["judge_prompt_config"],
            citation_judge_prompt_config=section.get("citation_judge_prompt_config"),
            relevance_judge_prompt_config=section.get("relevance_judge_prompt_config"),
            output_dir=out_dir,
            limit=section.get("limit"),
            overwrite=True,
            resume=bool(section.get("resume")),
            mode=section.get("mode") or "both",
            judge_provider=section.get("judge_provider"),
            judge_model=section.get("judge_model"),
            max_claims_per_answer=int(section.get("max_claims_per_answer") or 20),
            judge_max_retries=int(section.get("judge_max_retries") or 1),
        )

        
        eval_metrics_path = Path(info["metrics_json"])
        if eval_metrics_path.exists():
            payload = read_json(eval_metrics_path)
            mirror_json = metrics_dir / "llm_eval_metrics.json"
            write_json(mirror_json, payload)
            mirror_csv = metrics_dir / "llm_eval_metrics.csv"
            src_csv = Path(info["metrics_csv"])
            if src_csv.exists():
                mirror_csv.write_text(src_csv.read_text(encoding="utf-8"), encoding="utf-8")

            rag = (payload.get("rag") or {})
            cmp_ = (payload.get("comparison") or {})
            key_metrics = {
                "rag_faithfulness_soft": rag.get("faithfulness_soft"),
                "rag_hallucination_rate": rag.get("hallucination_rate"),
                "rag_citation_accuracy_rate": rag.get("citation_accuracy_rate"),
                "rag_citation_coverage_claim_level": rag.get("citation_coverage_claim_level"),
                "faithfulness_soft_improvement_abs": cmp_.get("faithfulness_soft_improvement_abs"),
                "hallucination_rate_reduction_abs": cmp_.get("hallucination_rate_reduction_abs"),
                "rag_meets_faithfulness_target": cmp_.get("rag_meets_faithfulness_target"),
                "rag_meets_hallucination_target": cmp_.get("rag_meets_hallucination_target"),
                "rag_meets_citation_accuracy_target": cmp_.get("rag_meets_citation_accuracy_target"),
                "rag_meets_faithfulness_improvement_target": cmp_.get("rag_meets_faithfulness_improvement_target"),
            }
        else:
            key_metrics = {}

        files = [
            info.get("metrics_json"),
            info.get("metrics_csv"),
            info.get("summary_csv"),
            info.get("summary_json"),
            info.get("claim_evaluations"),
            info.get("case_evaluations"),
            info.get("failed_judge_cases"),
            info.get("report_md"),
        ]
        files = [f for f in files if f]
        status.step_complete(name, key_metrics=key_metrics, output_files=files)
    except Exception as e:  
        logger.exception("llm_evaluation failed")
        status.step_fail(name, error=e)




def run(cfg: Dict[str, Any], *, log_path: Optional[Path] = None) -> Dict[str, Any]:
    paths = resolve_experiment_paths(cfg)
    out_dir = Path(paths["output_dir"])
    ensure_dir(out_dir)
    ensure_dir(Path(paths["metrics"]["dir"]))
    ensure_dir(Path(paths["summary"]["dir"]))

    
    Path(paths["config_resolved"]).write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    
    try:
        write_datasphere_commands(cfg, paths["datasphere_commands"])
    except Exception:  
        logger.exception("Could not render datasphere commands; continuing")

    status = PipelineStatus(
        experiment_name=cfg["experiment"]["name"],
        path=Path(paths["metrics"]["pipeline_status"]),
    )

    try:
        step_retrieval(cfg, paths, status)
        step_rerank(cfg, paths, status)
        step_generation(cfg, paths, status)
        step_comparison(cfg, paths, status)
        step_llm_evaluation(cfg, paths, status)
    finally:
        status.finalize()
        try:
            collect_summary(out_dir, cfg=cfg, pipeline_status=status.data)
        except Exception:  
            logger.exception("collect_summary failed")

    return status.data




def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiment pipeline")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output-dir", default=None, type=str)
    p.add_argument("--limit", default=None, type=int)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--skip-retrieval", action="store_true", default=False)
    p.add_argument("--skip-rerank", action="store_true", default=False)
    p.add_argument("--skip-generation", action="store_true", default=False)
    p.add_argument("--skip-comparison", action="store_true", default=False)
    p.add_argument("--skip-llm-eval", action="store_true", default=False)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_experiment_config(args.config)
    cfg = apply_cli_overrides(
        cfg,
        {
            "output_dir": args.output_dir,
            "limit": args.limit,
            "device": args.device,
            "overwrite": args.overwrite or None,
            "skip_retrieval": args.skip_retrieval,
            "skip_rerank": args.skip_rerank,
            "skip_generation": args.skip_generation,
            "skip_comparison": args.skip_comparison,
            "skip_llm_eval": args.skip_llm_eval,
        },
    )
    validate_experiment_config(cfg)
    assert_overwrite_allowed(cfg)

    paths = resolve_experiment_paths(cfg)
    out_dir = Path(paths["output_dir"])
    ensure_dir(out_dir)
    setup_logging(
        verbose=args.verbose,
        log_file=Path(paths["log_file"]),
        name="run_experiment",
    )
    logger.info("Experiment: %s", cfg["experiment"]["name"])
    logger.info("Output dir: %s", out_dir)

    status = run(cfg, log_path=Path(paths["log_file"]))

    overall = status.get("status", "unknown")
    logger.info("Final status: %s", overall)
    return 0 if overall in ("completed", "completed_with_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
