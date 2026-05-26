from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sys
from tqdm import tqdm

from .embedding_models import EmbeddingModel, ModelConfig, load_embedding_model
from .io_utils import (
    ensure_dir,
    list_model_keys,
    load_model_config,
    read_json,
    read_jsonl_list,
    setup_logging,
    write_json,
    write_jsonl,
)
from .metrics import aggregate_metrics, evaluate_query
from .report import (
    select_best_model,
    write_best_model,
    write_comparison_csv,
    write_comparison_json,
)
from .retrieval import NumpyRetriever

logger = logging.getLogger(__name__)


def _load_artifacts(model_dir: Path, model_key: str) -> tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    emb_path = model_dir / "embeddings.npy"
    meta_path = model_dir / "metadata.jsonl"
    info_path = model_dir / "run_info.json"

    for p in (emb_path, meta_path, info_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    embeddings = np.load(emb_path)
    metadata = read_jsonl_list(meta_path)
    run_info = read_json(info_path)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"[{model_key}] embeddings rows ({embeddings.shape[0]}) "
            f"!= metadata rows ({len(metadata)})"
        )
    declared_dim = int(run_info.get("embedding_dim", -1))
    if declared_dim != int(embeddings.shape[1]):
        raise ValueError(
            f"[{model_key}] embedding_dim mismatch: run_info={declared_dim} "
            f"actual={embeddings.shape[1]}"
        )
    declared_key = run_info.get("model_key")
    if declared_key != model_key:
        raise ValueError(
            f"[{model_key}] run_info.model_key='{declared_key}' does not match folder model_key"
        )
    return embeddings.astype(np.float32, copy=False), metadata, run_info


def _detailed_record(
    *,
    query_id: Any,
    query_text: str,
    model_key: str,
    expected_doc_ids: List[str],
    expected_section_keywords: Optional[List[str]],
    hits,
    metrics: Dict[str, Any],
    query_time_ms: float,
    retrieval_time_ms: float,
    text_preview_chars: int = 300,
) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query_text,
        "model_key": model_key,
        "expected_document_ids": expected_doc_ids,
        "expected_section_keywords": expected_section_keywords or [],
        "top_results": [h.to_dict(text_preview_chars=text_preview_chars) for h in hits],
        "hit_at_1": bool(metrics.get("hit_at_1", False)),
        "hit_at_5": bool(metrics.get("hit_at_5", False)),
        "hit_at_10": bool(metrics.get("hit_at_10", False)),
        "recall_at_5": float(metrics.get("recall_at_5", 0.0)),
        "precision_at_5": float(metrics.get("precision_at_5", 0.0)),
        "mrr": float(metrics.get("mrr", 0.0)),
        "first_relevant_rank": metrics.get("first_relevant_rank"),
        "section_keyword_hit_at_5": metrics.get("section_keyword_hit_at_5"),
        "query_time_ms": round(query_time_ms, 3),
        "retrieval_time_ms": round(retrieval_time_ms, 3),
    }


def evaluate_one_model(
    *,
    model_key: str,
    config_path: str | Path,
    embeddings_root: str | Path,
    queries: List[Dict[str, Any]],
    output_dir: str | Path,
    device: str = "auto",
    top_k: int = 10,
    text_preview_chars: int = 300,
) -> Dict[str, Any]:
    model_dir = Path(embeddings_root) / model_key
    embeddings, metadata, run_info = _load_artifacts(model_dir, model_key)
    logger.info(
        "[%s] Loaded %d embeddings (dim=%d) from %s",
        model_key,
        embeddings.shape[0],
        embeddings.shape[1],
        model_dir,
    )

    model_cfg = load_model_config(config_path, model_key)
    if bool(model_cfg.get("normalize")) != bool(run_info.get("normalize")):
        logger.warning(
            "[%s] normalize flag differs: config=%s run_info=%s. "
            "Using run_info for index, config for query encoding.",
            model_key,
            model_cfg.get("normalize"),
            run_info.get("normalize"),
        )

    model: EmbeddingModel = load_embedding_model(model_cfg, device=device)
    retriever = NumpyRetriever(
        embeddings,
        metadata,
        already_normalized=bool(run_info.get("normalize", True)),
    )

    per_query: List[Dict[str, Any]] = []
    detailed_records: List[Dict[str, Any]] = []
    query_times: List[float] = []
    retrieval_times: List[float] = []

    for q in tqdm(
        queries,
        desc=f"queries:{model_key}",
        file=sys.stdout,
        mininterval=0.5,
        dynamic_ncols=True,
    ):
        query_id = q.get("query_id")
        query_text = q.get("query") or ""
        expected_doc_ids = list(q.get("expected_document_ids") or [])
        expected_keywords = list(q.get("expected_section_keywords") or []) or None

        t0 = time.perf_counter()
        q_vec = model.encode_queries([query_text], show_progress_bar=False)[0]
        query_time_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        hits = retriever.search(q_vec, top_k=top_k)
        retrieval_time_ms = (time.perf_counter() - t1) * 1000.0

        metrics = evaluate_query(
            hits,
            expected_document_ids=expected_doc_ids,
            expected_section_keywords=expected_keywords,
            ks=(1, 5, 10),
        )
        per_query.append(metrics)
        query_times.append(query_time_ms)
        retrieval_times.append(retrieval_time_ms)

        detailed_records.append(
            _detailed_record(
                query_id=query_id,
                query_text=query_text,
                model_key=model_key,
                expected_doc_ids=expected_doc_ids,
                expected_section_keywords=expected_keywords,
                hits=hits,
                metrics=metrics,
                query_time_ms=query_time_ms,
                retrieval_time_ms=retrieval_time_ms,
            )
        )

    detailed_dir = ensure_dir(Path(output_dir) / "retrieval_results")
    detailed_path = detailed_dir / f"{model_key}_detailed_results.jsonl"
    write_jsonl(detailed_path, detailed_records)
    logger.info("[%s] Detailed results: %s", model_key, detailed_path)

    agg = aggregate_metrics(per_query)
    avg_q = sum(query_times) / len(query_times) if query_times else 0.0
    avg_r = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0

    row: Dict[str, Any] = {
        "model_key": model_key,
        "model_name": run_info.get("model_name"),
        "embedding_dim": int(run_info.get("embedding_dim", embeddings.shape[1])),
        "number_of_queries": len(queries),
        "hit_at_1": round(agg.get("hit_at_1", 0.0), 4),
        "hit_at_5": round(agg.get("hit_at_5", 0.0), 4),
        "hit_at_10": round(agg.get("hit_at_10", 0.0), 4),
        "recall_at_5": round(agg.get("recall_at_5", 0.0), 4),
        "precision_at_5": round(agg.get("precision_at_5", 0.0), 4),
        "mrr": round(agg.get("mrr", 0.0), 4),
        "avg_query_time_ms": round(avg_q, 3),
        "avg_retrieval_time_ms": round(avg_r, 3),
    }
    if "section_keyword_hit_at_5" in agg:
        row["section_keyword_hit_at_5"] = round(agg["section_keyword_hit_at_5"], 4)
        row["section_keyword_coverage"] = round(agg["section_keyword_coverage"], 4)
    return row


def evaluate_models(
    *,
    queries_path: str | Path,
    config_path: str | Path,
    embeddings_root: str | Path,
    model_keys: Optional[Sequence[str]],
    output_dir: str | Path,
    device: str = "auto",
    top_k: int = 10,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    ensure_dir(output_dir / "retrieval_results")

    queries = read_jsonl_list(queries_path)
    logger.info("Loaded %d evaluation queries from %s", len(queries), queries_path)

    if not model_keys:
        model_keys = list_model_keys(config_path)
        logger.info("No --model-keys provided, evaluating all from config: %s", list(model_keys))

    rows: List[Dict[str, Any]] = []
    for mk in model_keys:
        try:
            row = evaluate_one_model(
                model_key=mk,
                config_path=config_path,
                embeddings_root=embeddings_root,
                queries=queries,
                output_dir=output_dir,
                device=device,
                top_k=top_k,
            )
            rows.append(row)
            logger.info(
                "[%s] recall@5=%.3f precision@5=%.3f mrr=%.3f hit@1=%.3f hit@5=%.3f hit@10=%.3f",
                mk,
                row["recall_at_5"],
                row["precision_at_5"],
                row["mrr"],
                row["hit_at_1"],
                row["hit_at_5"],
                row["hit_at_10"],
            )
        except FileNotFoundError as e:
            logger.error("[%s] Missing artifacts: %s", mk, e)
        except Exception:
            logger.exception("[%s] Failed to evaluate", mk)

    if not rows:
        raise RuntimeError("No models were evaluated successfully")

    csv_path = reports_dir / "embedding_model_comparison.csv"
    json_path = reports_dir / "embedding_model_comparison.json"
    write_comparison_csv(rows, csv_path)
    write_comparison_json(rows, json_path)
    logger.info("Comparison saved: %s, %s", csv_path, json_path)

    best = select_best_model(rows)
    if best is not None:
        best_path = reports_dir / "best_embedding_model.json"
        write_best_model(best, best_path)
        logger.info("Best model: %s (%s)", best["best_model_key"], best["reason"])
        logger.info("Best metrics: %s", best["metrics"])

    return {"rows": rows, "best": best}


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality of embedding models")
    parser.add_argument("--queries-path", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--embeddings-root",
        required=True,
        type=str,
        help="Root dir that contains <model_key>/{embeddings.npy,metadata.jsonl,run_info.json}",
    )
    parser.add_argument(
        "--model-keys",
        nargs="+",
        default=None,
        help="Subset of model_keys to evaluate. If omitted, evaluates all from config.",
    )
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], type=str)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        verbose=args.verbose,
        log_file=out_dir / "evaluate_models.log",
        name="evaluate_models",
    )
    logger.info("Args: %s", vars(args))
    try:
        evaluate_models(
            queries_path=args.queries_path,
            config_path=args.config,
            embeddings_root=args.embeddings_root,
            model_keys=args.model_keys,
            output_dir=args.output_dir,
            device=args.device,
            top_k=args.top_k,
        )
    except Exception:
        logger.exception("Evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
