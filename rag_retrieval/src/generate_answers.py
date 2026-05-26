from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from .io_utils import ensure_dir, read_jsonl_list, setup_logging
from .llm_client import LLMClient
from .prompt_templates import load_prompt_config
from .rag_generation import (
    RetrievalEngine,
    generate_no_rag_answer,
    generate_rag_answer,
    load_cases,
)

logger = logging.getLogger(__name__)


def _existing_case_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen: Set[str] = set()
    try:
        for rec in read_jsonl_list(path):
            cid = rec.get("case_id")
            if cid:
                seen.add(str(cid))
    except Exception as e:
        logger.warning("Could not parse existing %s: %s. Treating as empty.", path, e)
    return seen


def _open_output(path: Path, *, overwrite: bool, resume: bool) -> Any:
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

def run_rag(
    *,
    cases: List[Dict[str, Any]],
    output_path: Path,
    llm_client: LLMClient,
    engine: RetrievalEngine,
    overwrite: bool,
    resume: bool,
    prompt_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    skip_ids = _existing_case_ids(output_path) if resume else set()
    if skip_ids:
        logger.info("--resume: skipping %d already-processed cases", len(skip_ids))

    fh = _open_output(output_path, overwrite=overwrite, resume=resume)
    n_ok = 0
    n_err = 0
    try:
        for c in tqdm(cases, desc="rag", file=sys.stdout, dynamic_ncols=True):
            case_id = str(c["case_id"])
            if case_id in skip_ids:
                continue
            try:
                rec = generate_rag_answer(
                    case_id=case_id,
                    patient_case=c["patient_case"],
                    engine=engine,
                    llm_client=llm_client,
                    extra_meta={"raw_query_fields": _safe_extra(c.get("raw"))},
                    prompt_config=prompt_config,
                )
                _write_record(fh, rec)
                if rec.get("answer_json"):
                    n_ok += 1
                else:
                    n_err += 1
            except Exception as e:
                logger.exception("Case %s failed", case_id)
                _write_record(fh, {
                    "case_id": case_id,
                    "patient_case": c["patient_case"],
                    "mode": "rag",
                    "answer_json": None,
                    "errors": [f"unexpected: {e}"],
                })
                n_err += 1
    finally:
        fh.close()
    logger.info("RAG generation done: ok=%d errors=%d -> %s", n_ok, n_err, output_path)
    return {
        "ok": n_ok,
        "errors": n_err,
        "num_processed": n_ok + n_err,
        "output_path": str(output_path),
    }


def run_no_rag(
    *,
    cases: List[Dict[str, Any]],
    output_path: Path,
    llm_client: LLMClient,
    overwrite: bool,
    resume: bool,
    prompt_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    skip_ids = _existing_case_ids(output_path) if resume else set()
    if skip_ids:
        logger.info("--resume: skipping %d already-processed cases", len(skip_ids))

    fh = _open_output(output_path, overwrite=overwrite, resume=resume)
    n_ok = 0
    n_err = 0
    try:
        for c in tqdm(cases, desc="no_rag", file=sys.stdout, dynamic_ncols=True):
            case_id = str(c["case_id"])
            if case_id in skip_ids:
                continue
            try:
                rec = generate_no_rag_answer(
                    case_id=case_id,
                    patient_case=c["patient_case"],
                    llm_client=llm_client,
                    extra_meta={"raw_query_fields": _safe_extra(c.get("raw"))},
                    prompt_config=prompt_config,
                )
                _write_record(fh, rec)
                if rec.get("answer_json"):
                    n_ok += 1
                else:
                    n_err += 1
            except Exception as e:
                logger.exception("Case %s failed (no_rag)", case_id)
                _write_record(fh, {
                    "case_id": case_id,
                    "patient_case": c["patient_case"],
                    "mode": "no_rag",
                    "answer_json": None,
                    "errors": [f"unexpected: {e}"],
                })
                n_err += 1
    finally:
        fh.close()
    logger.info("No-RAG generation done: ok=%d errors=%d -> %s", n_ok, n_err, output_path)
    return {
        "ok": n_ok,
        "errors": n_err,
        "num_processed": n_ok + n_err,
        "output_path": str(output_path),
    }

def run_generation(
    *,
    mode: str,
    cases_path: str | Path,
    output_path: str | Path,
    llm_config: str | Path,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    prompt_config_path: Optional[str | Path] = None,
    case_id_field: Optional[str] = None,
    patient_case_field: Optional[str] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = False,
    embedding_model_key: Optional[str] = None,
    embedding_config: Optional[str | Path] = None,
    embeddings_dir: Optional[str | Path] = None,
    reranker_key: Optional[str] = None,
    reranker_config: Optional[str | Path] = None,
    candidate_top_k: int = 30,
    final_top_k: int = 5,
    context_selection: str = "anchor_page",
    context_page_tolerance: int = 1,
    device: str = "auto",
) -> Dict[str, Any]:
    if mode not in ("rag", "no_rag"):
        raise ValueError(f"mode must be 'rag' or 'no_rag', got {mode!r}")

    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    cases = load_cases(
        cases_path,
        case_id_field=case_id_field,
        patient_case_field=patient_case_field,
    )
    if limit is not None and limit > 0:
        cases = cases[:limit]
    if not cases:
        raise RuntimeError(f"No cases loaded from {cases_path}")

    prompt_config = load_prompt_config(prompt_config_path) if prompt_config_path else None
    llm_client = LLMClient(str(llm_config), provider=llm_provider, model_name=llm_model)

    if mode == "rag":
        for name, val in (
            ("embedding_model_key", embedding_model_key),
            ("embedding_config", embedding_config),
            ("embeddings_dir", embeddings_dir),
            ("reranker_key", reranker_key),
            ("reranker_config", reranker_config),
        ):
            if val is None:
                raise ValueError(f"{name} is required for mode=rag")
        engine = RetrievalEngine.from_config(
            embeddings_dir=str(embeddings_dir),
            embedding_model_key=str(embedding_model_key),
            embedding_config=str(embedding_config),
            reranker_key=str(reranker_key),
            reranker_config=str(reranker_config),
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
            context_selection=context_selection,
            context_page_tolerance=context_page_tolerance,
            device=device,
        )
        return run_rag(
            cases=cases,
            output_path=out_path,
            llm_client=llm_client,
            engine=engine,
            overwrite=overwrite,
            resume=resume,
            prompt_config=prompt_config,
        )

    return run_no_rag(
        cases=cases,
        output_path=out_path,
        llm_client=llm_client,
        overwrite=overwrite,
        resume=resume,
        prompt_config=prompt_config,
    )


def _safe_extra(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    keep = (
        "query_id",
        "query_type",
        "difficulty",
        "expected_document_ids",
        "expected_chunk_ids",
        "expected_section_ids",
        "expected_section_titles",
        "expected_labels",
        "expected_specialties",
        "ground_truth_diagnoses",
    )
    return {k: raw.get(k) for k in keep if k in raw}


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate answers for clinical cases (RAG / no-RAG)")
    p.add_argument("--cases-path", required=True, type=str)
    p.add_argument("--mode", choices=["rag", "no_rag"], required=True)
    p.add_argument("--llm-config", required=True, type=str)
    p.add_argument("--llm-provider", default=None, type=str,
                   help="Override llm.provider from config (e.g. mock).")
    p.add_argument("--llm-model", default=None, type=str,
                   help="Override llm.model_name from config.")

    p.add_argument(
        "--prompt-config",
        default=None,
        type=str,
        help="Путь к YAML prompt-config'у. Если не задан, используется fallback "
             "из prompt_templates.py.",
    )
    p.add_argument(
        "--case-id-field",
        default=None,
        type=str,
        help="Имя поля в JSONL для case_id (по умолчанию case_id → query_id).",
    )
    p.add_argument(
        "--patient-case-field",
        default=None,
        type=str,
        help="Имя поля в JSONL для patient_case (по умолчанию patient_case → query).",
    )

    p.add_argument("--embedding-model-key", default=None, type=str)
    p.add_argument("--embedding-config", default=None, type=str)
    p.add_argument("--embeddings-dir", default=None, type=str)
    p.add_argument("--reranker-key", default=None, type=str)
    p.add_argument("--reranker-config", default=None, type=str)
    p.add_argument("--candidate-top-k", type=int, default=30)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--context-selection", default="anchor_page", type=str)
    p.add_argument("--context-page-tolerance", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], type=str)

    p.add_argument("--output-path", required=True, type=str)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)

    log_path = out_path.with_suffix(out_path.suffix + ".log")
    setup_logging(verbose=args.verbose, log_file=log_path, name=f"gen-{args.mode}")
    logger.info("Args: %s", vars(args))

    try:
        run_generation(
            mode=args.mode,
            cases_path=args.cases_path,
            output_path=out_path,
            llm_config=args.llm_config,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            prompt_config_path=args.prompt_config,
            case_id_field=args.case_id_field,
            patient_case_field=args.patient_case_field,
            limit=args.limit,
            overwrite=args.overwrite,
            resume=args.resume,
            embedding_model_key=args.embedding_model_key,
            embedding_config=args.embedding_config,
            embeddings_dir=args.embeddings_dir,
            reranker_key=args.reranker_key,
            reranker_config=args.reranker_config,
            candidate_top_k=args.candidate_top_k,
            final_top_k=args.final_top_k,
            context_selection=args.context_selection,
            context_page_tolerance=args.context_page_tolerance,
            device=args.device,
        )
    except FileExistsError as e:
        logger.error("%s", e)
        return 2
    except (ValueError, RuntimeError) as e:
        logger.error("%s", e)
        return 2
    except Exception:
        logger.exception("Generation failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
