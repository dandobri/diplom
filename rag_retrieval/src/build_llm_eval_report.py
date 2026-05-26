from __future__ import annotations

import argparse
import collections
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .io_utils import read_json, read_jsonl_list
from .llm_eval_metrics import (
    TARGET_CITATION_ACCURACY,
    TARGET_FAITHFULNESS_IMPROVEMENT_ABS,
    TARGET_FAITHFULNESS_RAG,
    TARGET_HALLUCINATION_RAG,
    TARGET_HALLUCINATION_REDUCTION_ABS,
)
from .prompt_templates import load_prompt_config

logger = logging.getLogger(__name__)


def _pretty(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _check(v: Optional[bool]) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    return "—"


def _safe_load_prompt(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        return load_prompt_config(path)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load prompt config %s: %s", path, e)
        return None


def _section_setup(metrics: Dict[str, Any]) -> List[str]:
    meta = metrics.get("meta") or {}
    lines = ["## 1. Experiment setup\n"]
    rows = [
        ("Started at", meta.get("started_at")),
        ("Finished at", meta.get("finished_at")),
        ("Elapsed (sec)", meta.get("elapsed_sec")),
        ("Judge LLM provider", meta.get("judge_provider")),
        ("Judge LLM provider type", meta.get("judge_provider_type")),
        ("Judge LLM model", meta.get("judge_model")),
        ("Faithfulness judge config", meta.get("judge_prompt_config")),
        ("Citation judge config", meta.get("citation_judge_prompt_config")),
        ("Relevance judge config", meta.get("relevance_judge_prompt_config")),
        ("Mode", meta.get("mode")),
        ("Limit", meta.get("limit")),
        ("Max claims per answer", meta.get("max_claims_per_answer")),
        ("Pairs input", meta.get("num_pairs_input")),
        ("Case evaluations total", meta.get("num_case_evaluations_total")),
    ]
    lines.append("| Поле | Значение |")
    lines.append("|---|---|")
    for k, v in rows:
        lines.append(f"| {k} | {_pretty(v)} |")
    lines.append("")
    return lines


def _section_datasets(
    metrics: Dict[str, Any],
    case_evaluations: Sequence[Dict[str, Any]],
    cases_meta: Dict[str, Dict[str, Any]],
) -> List[str]:
    meta = metrics.get("meta") or {}
    lines = ["## 2. Datasets\n"]
    lines.append(f"- Pairs path: `{meta.get('pairs_path') or '—'}`")
    lines.append(f"- Clinical cases path: `{meta.get('clinical_cases_path') or '—'}`")
    lines.append(f"- Total cases evaluated: **{metrics.get('num_cases')}**\n")

    if cases_meta:
        diff_counter = collections.Counter()
        type_counter = collections.Counter()
        evaluated_ids = {ce.get("case_id") for ce in case_evaluations if ce.get("case_id")}
        for cid in evaluated_ids:
            m = cases_meta.get(cid) or {}
            diff_counter[m.get("difficulty") or "unknown"] += 1
            type_counter[m.get("case_type") or "unknown"] += 1
        if diff_counter:
            lines.append("**Distribution by difficulty:**\n")
            lines.append("| Difficulty | Count |")
            lines.append("|---|---|")
            for k, v in sorted(diff_counter.items(), key=lambda x: -x[1]):
                lines.append(f"| {k} | {v} |")
            lines.append("")
        if type_counter:
            lines.append("**Distribution by case_type:**\n")
            lines.append("| case_type | Count |")
            lines.append("|---|---|")
            for k, v in sorted(type_counter.items(), key=lambda x: -x[1]):
                lines.append(f"| {k} | {v} |")
            lines.append("")
    return lines


def _section_prompt(title: str, idx: str, prompt_cfg: Optional[Dict[str, Any]], path: Optional[str]) -> List[str]:
    lines = [f"## {idx}. {title}\n"]
    if not prompt_cfg:
        lines.append(f"_Prompt config not loaded (path={path})._\n")
        return lines
    lines.append(f"- Path: `{path}`")
    lines.append(f"- Name: `{prompt_cfg.get('name')}`")
    lines.append(f"- Version: `{prompt_cfg.get('version')}`")
    lines.append(f"- Language: `{prompt_cfg.get('language')}`")
    purp = prompt_cfg.get("purpose")
    if purp:
        lines.append(f"- Purpose: {purp}")
    rules = prompt_cfg.get("rules") or []
    if rules:
        lines.append("- Rules:")
        for r in rules[:8]:
            lines.append(f"  - {r}")
    lines.append("")
    return lines


def _section_main_metrics(metrics: Dict[str, Any]) -> List[str]:
    rag = metrics.get("rag") or {}
    no_rag = metrics.get("no_rag") or {}
    cmp_ = metrics.get("comparison") or {}
    lines = ["## 6. Main metrics (RAG vs no-RAG)\n"]
    rows = [
        ("num_cases", rag.get("num_cases"), no_rag.get("num_cases")),
        ("valid_json_count", rag.get("valid_json_count"), no_rag.get("valid_json_count")),
        ("valid_json_rate", rag.get("valid_json_rate"), no_rag.get("valid_json_rate")),
        ("total_claims", rag.get("total_claims"), no_rag.get("total_claims")),
        ("supported_claims", rag.get("total_supported_claims"), no_rag.get("total_supported_claims")),
        ("partially_supported_claims", rag.get("total_partially_supported_claims"), no_rag.get("total_partially_supported_claims")),
        ("unsupported_claims", rag.get("total_unsupported_claims"), no_rag.get("total_unsupported_claims")),
        ("contradicted_claims", rag.get("total_contradicted_claims"), no_rag.get("total_contradicted_claims")),
        ("faithfulness_strict", rag.get("faithfulness_strict"), no_rag.get("faithfulness_strict")),
        ("faithfulness_soft", rag.get("faithfulness_soft"), no_rag.get("faithfulness_soft")),
        ("hallucination_rate", rag.get("hallucination_rate"), no_rag.get("hallucination_rate")),
        ("answer_relevance_avg", rag.get("answer_relevance_avg"), no_rag.get("answer_relevance_avg")),
        ("clinical_usefulness_avg", rag.get("clinical_usefulness_avg"), no_rag.get("clinical_usefulness_avg")),
        ("safety_avg", rag.get("safety_avg"), no_rag.get("safety_avg")),
        ("top1_diagnosis_match_rate", rag.get("top1_diagnosis_match_rate"), no_rag.get("top1_diagnosis_match_rate")),
        ("top3_diagnosis_match_rate", rag.get("top3_diagnosis_match_rate"), no_rag.get("top3_diagnosis_match_rate")),
    ]
    lines.append("| Метрика | RAG | no-RAG |")
    lines.append("|---|---|---|")
    for name, r, n in rows:
        lines.append(f"| {name} | {_pretty(r)} | {_pretty(n)} |")
    lines.append("")

    lines.append("**Citation metrics (RAG only):**\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    for k in ("citation_coverage_item_level", "citation_coverage_claim_level",
              "citation_validity_rate", "citation_accuracy_rate",
              "invalid_citation_count_total"):
        lines.append(f"| {k} | {_pretty(rag.get(k))} |")
    lines.append("")

    lines.append("**Comparison:**\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    for k in ("faithfulness_soft_improvement_abs", "faithfulness_soft_improvement_rel",
              "hallucination_rate_reduction_abs",
              "case_level_wins_rag_better", "case_level_losses_rag_worse",
              "case_level_ties"):
        lines.append(f"| {k} | {_pretty(cmp_.get(k))} |")
    lines.append("")
    return lines


def _section_case_compare(summary_rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["## 7. Where RAG > no-RAG / RAG < no-RAG (case-level)\n"]
    if not summary_rows:
        lines.append("_No summary rows._\n")
        return lines

    def _delta(r):
        d = r.get("faithfulness_improvement_abs")
        return d if isinstance(d, (int, float)) else None

    with_delta = [r for r in summary_rows if _delta(r) is not None]
    with_delta.sort(key=lambda r: _delta(r), reverse=True)

    top_rag_better = with_delta[:5]
    top_rag_worse = list(reversed(with_delta[-5:])) if len(with_delta) >= 5 else list(reversed(with_delta))

    lines.append("**Top-5 cases where RAG > no-RAG (по Δ faithfulness_soft):**\n")
    lines.append("| case_id | difficulty | RAG soft | no-RAG soft | Δ |")
    lines.append("|---|---|---|---|---|")
    for r in top_rag_better:
        lines.append(
            f"| {r.get('case_id')} | {_pretty(r.get('difficulty'))} | "
            f"{_pretty(r.get('rag_faithfulness_soft'))} | {_pretty(r.get('no_rag_faithfulness_soft'))} | "
            f"{_pretty(_delta(r))} |"
        )
    lines.append("")

    lines.append("**Top-5 cases where RAG < no-RAG (по Δ faithfulness_soft):**\n")
    lines.append("| case_id | difficulty | RAG soft | no-RAG soft | Δ |")
    lines.append("|---|---|---|---|---|")
    for r in top_rag_worse:
        lines.append(
            f"| {r.get('case_id')} | {_pretty(r.get('difficulty'))} | "
            f"{_pretty(r.get('rag_faithfulness_soft'))} | {_pretty(r.get('no_rag_faithfulness_soft'))} | "
            f"{_pretty(_delta(r))} |"
        )
    lines.append("")
    return lines


def _section_targets(metrics: Dict[str, Any]) -> List[str]:
    cmp_ = metrics.get("comparison") or {}
    rag = metrics.get("rag") or {}
    lines = ["## 8. Target thresholds (ВКР)\n"]
    rows = [
        ("RAG faithfulness_soft >= "
         f"{TARGET_FAITHFULNESS_RAG}",
         rag.get("faithfulness_soft"),
         cmp_.get("rag_meets_faithfulness_target")),
        ("RAG hallucination_rate < "
         f"{TARGET_HALLUCINATION_RAG}",
         rag.get("hallucination_rate"),
         cmp_.get("rag_meets_hallucination_target")),
        ("RAG citation_accuracy_rate >= "
         f"{TARGET_CITATION_ACCURACY}",
         rag.get("citation_accuracy_rate"),
         cmp_.get("rag_meets_citation_accuracy_target")),
        ("Faithfulness_soft improvement (RAG − no-RAG) >= "
         f"{TARGET_FAITHFULNESS_IMPROVEMENT_ABS}",
         cmp_.get("faithfulness_soft_improvement_abs"),
         cmp_.get("rag_meets_faithfulness_improvement_target")),
        ("Hallucination_rate reduction (no-RAG − RAG) >= "
         f"{TARGET_HALLUCINATION_REDUCTION_ABS}",
         cmp_.get("hallucination_rate_reduction_abs"),
         cmp_.get("rag_meets_hallucination_reduction_target")),
    ]
    lines.append("| Цель | Значение | Pass/Fail |")
    lines.append("|---|---|---|")
    for label, value, status in rows:
        lines.append(f"| {label} | {_pretty(value)} | {_check(status)} |")
    lines.append("")
    overall = all(s is True for _, _, s in rows if s is not None)
    has_any = any(s is True for _, _, s in rows)
    if overall and has_any:
        lines.append("**Итог: ВСЕ цели ВКР достигнуты ✓**\n")
    elif has_any:
        lines.append("**Итог: часть целей не достигнута. См. детальный анализ ошибок ниже.**\n")
    else:
        lines.append("**Итог: метрики недоступны или цели не оценены.**\n")
    return lines


def _section_failed(failed_path: Path) -> List[str]:
    lines = ["## 9. Failed judge cases\n"]
    if not failed_path.exists():
        lines.append("_No failed_judge_cases.jsonl found._\n")
        return lines
    try:
        rows = read_jsonl_list(failed_path)
    except Exception as e:  # noqa: BLE001
        lines.append(f"_Failed to read {failed_path}: {e}_\n")
        return lines
    if not rows:
        lines.append("_Список пуст — все judge-вызовы успешны._\n")
        return lines
    lines.append(f"Всего: **{len(rows)}** записей.\n")
    lines.append("| case_id | mode | errors |")
    lines.append("|---|---|---|")
    for r in rows[:20]:
        errs = r.get("errors") or []
        errs_str = "; ".join(str(e) for e in errs)[:200]
        lines.append(f"| {r.get('case_id')} | {r.get('mode')} | {errs_str} |")
    if len(rows) > 20:
        lines.append(f"\n_(показаны первые 20 из {len(rows)}.)_\n")
    lines.append("")
    return lines


def _section_error_analysis(case_evaluations: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["## 10. Error analysis\n"]
    rag_evals = [c for c in case_evaluations if c.get("mode") == "rag"]
    no_rag_evals = [c for c in case_evaluations if c.get("mode") == "no_rag"]

    def _stats(evals):
        total = sum(int((c.get("per_case") or {}).get("num_claims") or 0) for c in evals)
        unsup = sum(int((c.get("per_case") or {}).get("unsupported_claims") or 0) for c in evals)
        contra = sum(int((c.get("per_case") or {}).get("contradicted_claims") or 0) for c in evals)
        return total, unsup, contra

    rag_t, rag_u, rag_c = _stats(rag_evals)
    no_rag_t, no_rag_u, no_rag_c = _stats(no_rag_evals)

    lines.append("| Mode | total_claims | unsupported | contradicted |")
    lines.append("|---|---|---|---|")
    lines.append(f"| rag | {rag_t} | {rag_u} | {rag_c} |")
    lines.append(f"| no_rag | {no_rag_t} | {no_rag_u} | {no_rag_c} |")
    lines.append("")
    return lines


_LIMITATIONS_TEXT = """\
## 11. Limitations

- Это **LLM-as-a-judge** оценка. Она НЕ заменяет ручную экспертно-врачебную проверку.
- Часть кейсов в `data/clinical_cases_v1.jsonl` помечены как `auto_generated`
  (review_status) и нуждаются в ручной верификации формулировок и
  `expected_diagnoses`.
- `citation_validity_rate` — техническая метрика (совпадение source_id /
  chunk_id / document_id с retrieved_chunks). Семантическое подтверждение
  (`citation_accuracy_rate`) даётся judge-моделью.
- `faithfulness_*` и `hallucination_rate` зависят от качества reference_context.
  Для no-RAG в качестве reference берётся набор retrieved_chunks из парного
  RAG-запуска (apples-to-apples), но это не идеальный ground truth — часть
  правильных no-RAG-claims может быть помечена как `not_enough_information`,
  если они не попали в retrieval.
- Финальная экспертно-врачебная проверка ответов рекомендуется для
  пограничных или жизнеугрожающих сценариев.
"""


def build_report(
    *,
    output_dir: Path,
    metrics: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    case_evaluations: Sequence[Dict[str, Any]],
    cases_meta: Dict[str, Dict[str, Any]],
    failed_path: Path,
    judge_prompt_config_path: Optional[str],
    citation_judge_prompt_config_path: Optional[str],
    relevance_judge_prompt_config_path: Optional[str],
    rag_prompt_config_path: Optional[str] = None,
    no_rag_prompt_config_path: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    out = output_path or (Path(output_dir) / "llm_eval_report.md")
    lines: List[str] = []
    lines.append("# LLM evaluation report\n")
    lines.append(f"_Generated for output_dir=`{output_dir}`._\n")

    lines.extend(_section_setup(metrics))
    lines.extend(_section_datasets(metrics, case_evaluations, cases_meta))

    lines.extend(_section_prompt("RAG prompt", "3", _safe_load_prompt(rag_prompt_config_path), rag_prompt_config_path))
    lines.extend(_section_prompt("no-RAG prompt", "4", _safe_load_prompt(no_rag_prompt_config_path), no_rag_prompt_config_path))
    # 5. Judge prompts (3 sub-blocks).
    lines.append("## 5. Judge prompts\n")
    for sub_idx, (title, path) in enumerate(
        [
            ("Faithfulness judge", judge_prompt_config_path),
            ("Citation accuracy judge", citation_judge_prompt_config_path),
            ("Answer relevance judge", relevance_judge_prompt_config_path),
        ],
        start=1,
    ):
        cfg = _safe_load_prompt(path)
        lines.append(f"### 5.{sub_idx}. {title}")
        if not cfg:
            lines.append(f"_path={path or '—'} (config not loaded)_\n")
            continue
        lines.append(f"- Path: `{path}`")
        lines.append(f"- Name: `{cfg.get('name')}`")
        lines.append(f"- Version: `{cfg.get('version')}`")
        if cfg.get("purpose"):
            lines.append(f"- Purpose: {cfg['purpose']}")
        rules = cfg.get("rules") or []
        if rules:
            lines.append("- Rules:")
            for r in rules[:6]:
                lines.append(f"  - {r}")
        lines.append("")

    lines.extend(_section_main_metrics(metrics))
    lines.extend(_section_case_compare(summary_rows))
    lines.extend(_section_targets(metrics))
    lines.extend(_section_failed(failed_path))
    lines.extend(_section_error_analysis(case_evaluations))
    lines.append(_LIMITATIONS_TEXT)

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote LLM eval report → %s", out)
    return out


# ---------- CLI ----------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build llm_eval_report.md from llm_eval/ artifacts")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--rag-prompt-config", default=None, type=str)
    p.add_argument("--no-rag-prompt-config", default=None, type=str)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    metrics_path = out_dir / "llm_eval_metrics.json"
    cases_path = out_dir / "case_evaluations.jsonl"
    summary_csv = out_dir / "rag_vs_no_rag_eval_summary.csv"
    failed_path = out_dir / "failed_judge_cases.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")

    metrics = read_json(metrics_path)
    case_evaluations: List[Dict[str, Any]] = (
        list(read_jsonl_list(cases_path)) if cases_path.exists() else []
    )
    summary_rows: List[Dict[str, Any]] = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary_rows.append(dict(row))

    # Подгрузим cases_meta, если путь известен.
    cases_meta: Dict[str, Dict[str, Any]] = {}
    cm_path = (metrics.get("meta") or {}).get("clinical_cases_path")
    if cm_path and Path(cm_path).exists():
        for c in read_jsonl_list(cm_path):
            cid = c.get("case_id")
            if cid:
                cases_meta[str(cid)] = c

    meta = metrics.get("meta") or {}
    build_report(
        output_dir=out_dir,
        metrics=metrics,
        summary_rows=summary_rows,
        case_evaluations=case_evaluations,
        cases_meta=cases_meta,
        failed_path=failed_path,
        judge_prompt_config_path=meta.get("judge_prompt_config"),
        citation_judge_prompt_config_path=meta.get("citation_judge_prompt_config"),
        relevance_judge_prompt_config_path=meta.get("relevance_judge_prompt_config"),
        rag_prompt_config_path=args.rag_prompt_config,
        no_rag_prompt_config_path=args.no_rag_prompt_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
