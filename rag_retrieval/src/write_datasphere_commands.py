from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from .experiment_config import (
    apply_cli_overrides,
    load_experiment_config,
    resolve_experiment_paths,
    validate_experiment_config,
)
from .io_utils import ensure_dir


DEFAULT_TEMPLATE_PATH = Path("configs/experiments/datasphere_pipeline_commands_template.md")


def _g(cfg: Dict[str, Any], section: str, key: str, default: Any = "") -> Any:
    val = (cfg.get(section) or {}).get(key, default)
    return val if val is not None else default


def build_substitutions(cfg: Dict[str, Any]) -> Dict[str, str]:
    paths = resolve_experiment_paths(cfg)
    out_dir = paths["output_dir"]

    gen = cfg.get("generation") or {}
    cases_field = (gen.get("mode_cases_field") or {})

    rer = cfg.get("reranking_eval") or {}
    retr = cfg.get("retrieval_eval") or {}
    llm_eval = cfg.get("llm_evaluation") or {}

    embeddings_dir = (
        rer.get("embeddings_dir")
        or gen.get("embeddings_dir")
        or _g(cfg, "artifacts", "embeddings_dir", "")
    )
    embedding_model_key = (
        rer.get("embedding_model_key") or gen.get("embedding_model_key") or ""
    )
    embedding_config = (
        rer.get("embedding_config") or gen.get("embedding_config") or retr.get("config_path") or ""
    )
    reranker_key = rer.get("reranker_key") or gen.get("reranker_key") or ""
    reranker_config = rer.get("reranker_config") or gen.get("reranker_config") or ""
    candidate_top_k = rer.get("candidate_top_k") or gen.get("candidate_top_k") or 30
    final_top_k = rer.get("final_top_k") or gen.get("final_top_k") or 5
    context_selection = (
        rer.get("context_selection") or gen.get("context_selection") or "anchor_page"
    )
    device = rer.get("device") or gen.get("device") or "auto"

    model_keys = retr.get("model_keys") or []
    model_keys_str = " ".join(str(k) for k in model_keys) if model_keys else ""

    return {
        "{{EXPERIMENT_NAME}}": str(_g(cfg, "experiment", "name", "")),
        "{{OUTPUT_DIR}}": out_dir,
        "{{RETRIEVAL_QUERIES_PATH}}": str(_g(cfg, "data", "retrieval_queries_path", "")),
        "{{CLINICAL_CASES_PATH}}": str(_g(cfg, "data", "clinical_cases_path", "")),
        "{{EMBEDDING_CONFIG}}": str(embedding_config),
        "{{EMBEDDINGS_ROOT}}": str(_g(cfg, "artifacts", "embeddings_root", "")),
        "{{EMBEDDINGS_DIR}}": str(embeddings_dir),
        "{{EMBEDDING_MODEL_KEY}}": str(embedding_model_key),
        "{{MODEL_KEYS}}": model_keys_str,
        "{{TOP_K}}": str(retr.get("top_k") or 10),
        "{{RERANKER_KEY}}": str(reranker_key),
        "{{RERANKER_CONFIG}}": str(reranker_config),
        "{{CANDIDATE_TOP_K}}": str(candidate_top_k),
        "{{FINAL_TOP_K}}": str(final_top_k),
        "{{CONTEXT_SELECTION}}": str(context_selection),
        "{{DEVICE}}": str(device),
        "{{LLM_CONFIG}}": str(gen.get("llm_config") or ""),
        "{{PROMPT_CONFIG_RAG}}": str(gen.get("prompt_config_rag") or ""),
        "{{PROMPT_CONFIG_NO_RAG}}": str(gen.get("prompt_config_no_rag") or ""),
        "{{LIMIT}}": str(gen.get("limit") or 20),
        "{{CASE_ID_FIELD}}": str(cases_field.get("case_id_field") or "case_id"),
        "{{PATIENT_CASE_FIELD}}": str(cases_field.get("patient_case_field") or "patient_case"),
        "{{JUDGE_PROMPT_CONFIG}}": str(llm_eval.get("judge_prompt_config") or "configs/prompts/judge_faithfulness_v1.yaml"),
        "{{CITATION_JUDGE_PROMPT_CONFIG}}": str(llm_eval.get("citation_judge_prompt_config") or "configs/prompts/judge_citation_accuracy_v1.yaml"),
        "{{RELEVANCE_JUDGE_PROMPT_CONFIG}}": str(llm_eval.get("relevance_judge_prompt_config") or "configs/prompts/judge_answer_relevance_v1.yaml"),
        "{{LLM_EVAL_LIMIT}}": str(llm_eval.get("limit") or gen.get("limit") or 20),
        "{{LLM_EVAL_MODE}}": str(llm_eval.get("mode") or "both"),
        "{{LLM_EVAL_MAX_CLAIMS}}": str(llm_eval.get("max_claims_per_answer") or 20),
    }


def render_commands(
    cfg: Dict[str, Any],
    *,
    template_path: Optional[Path] = None,
) -> str:
    template = (template_path or DEFAULT_TEMPLATE_PATH).read_text(encoding="utf-8")
    subs = build_substitutions(cfg)
    text = template
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def write_commands(
    cfg: Dict[str, Any],
    output_path: str | Path,
    *,
    template_path: Optional[Path] = None,
) -> Path:
    out = Path(output_path)
    ensure_dir(out.parent)
    out.write_text(render_commands(cfg, template_path=template_path), encoding="utf-8")
    return out


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render datasphere_pipeline_commands.md from experiment YAML"
    )
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument(
        "--template",
        type=str,
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Путь к template-файлу (по умолчанию configs/experiments/datasphere_pipeline_commands_template.md).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_experiment_config(args.config)
    cfg = apply_cli_overrides(cfg, {})
    validate_experiment_config(cfg)
    out = write_commands(cfg, args.output, template_path=Path(args.template))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
