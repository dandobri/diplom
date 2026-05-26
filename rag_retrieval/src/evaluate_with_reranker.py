from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .advanced_metrics import aggregate_advanced, evaluate_query_advanced
from .context_selection import SUPPORTED_MODES, select_context_top_k
from .embedding_models import load_embedding_model
from .io_utils import (
    ensure_dir,
    load_model_config,
    read_json,
    read_jsonl_list,
    setup_logging,
    write_json,
    write_jsonl,
)
from .reranker import (
    Reranker,
    build_reranker,
    format_passage_for_reranker,
    load_reranker_config,
)
from .retrieval import NumpyRetriever, RetrievalHit

logger = logging.getLogger(__name__)


def _load_embedding_artifacts(
    embeddings_dir: Path, expected_model_key: str
) -> tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    emb_path = embeddings_dir / "embeddings.npy"
    meta_path = embeddings_dir / "metadata.jsonl"
    info_path = embeddings_dir / "run_info.json"
    for p in (emb_path, meta_path, info_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    embeddings = np.load(emb_path).astype(np.float32, copy=False)
    metadata = read_jsonl_list(meta_path)
    run_info = read_json(info_path)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != metadata rows ({len(metadata)})"
        )
    declared_dim = int(run_info.get("embedding_dim", -1))
    if declared_dim != int(embeddings.shape[1]):
        raise ValueError(
            f"embedding_dim mismatch: run_info={declared_dim} actual={embeddings.shape[1]}"
        )
    declared_key = run_info.get("model_key")
    if declared_key != expected_model_key:
        raise ValueError(
            f"run_info.model_key='{declared_key}' != --embedding-model-key='{expected_model_key}'"
        )
    return embeddings, metadata, run_info


def _hit_to_dense_dict(hit: RetrievalHit, *, text_preview_chars: int = 300) -> Dict[str, Any]:
    return {
        "rank": hit.rank,
        "dense_score": float(hit.score),
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "document_title": hit.document_title,
        "section_id": hit.section_id,
        "section_title": hit.section_title,
        "label": hit.label,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "text_preview": (hit.text or "")[:text_preview_chars],
    }


def _hit_to_reranked_dict(
    hit: RetrievalHit,
    *,
    new_rank: int,
    dense_rank: Optional[int],
    reranker_score: Optional[float],
    text_preview_chars: int = 300,
) -> Dict[str, Any]:
    d = _hit_to_dense_dict(hit, text_preview_chars=text_preview_chars)
    d["rank"] = new_rank
    d["dense_rank"] = int(dense_rank) if dense_rank is not None else None
    d["reranker_score"] = float(reranker_score) if reranker_score is not None else None
    if isinstance(hit.extra, dict) and "context_source" in hit.extra:
        d["context_source"] = hit.extra["context_source"]
    return d


def _rerank(
    hits: Sequence[RetrievalHit],
    query_text: str,
    metadata: Sequence[Dict[str, Any]],
    chunk_id_to_index: Dict[str, int],
    reranker: Reranker,
) -> tuple[List[RetrievalHit], List[float], List[int]]:
    if not hits:
        return [], [], []

    passages: List[str] = []
    for h in hits:
        meta_idx = chunk_id_to_index.get(h.chunk_id)
        meta = metadata[meta_idx] if meta_idx is not None else {}
        passages.append(format_passage_for_reranker(meta))

    scores = reranker.score(query_text, passages)
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)

    reranked: List[RetrievalHit] = []
    reranked_scores: List[float] = []
    dense_ranks: List[int] = []
    for new_rank, idx in enumerate(order, start=1):
        original_hit = hits[idx]
        new_hit = RetrievalHit(
            rank=new_rank,
            score=original_hit.score,
            chunk_id=original_hit.chunk_id,
            document_id=original_hit.document_id,
            document_title=original_hit.document_title,
            section_id=original_hit.section_id,
            section_title=original_hit.section_title,
            label=original_hit.label,
            source=original_hit.source,
            page_start=original_hit.page_start,
            page_end=original_hit.page_end,
            text=original_hit.text,
            extra=dict(original_hit.extra),
        )
        reranked.append(new_hit)
        reranked_scores.append(scores[idx])
        dense_ranks.append(original_hit.rank)
    return reranked, reranked_scores, dense_ranks


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _build_markdown_report(
    *,
    embedding_model_key: str,
    embedding_model_name: str,
    reranker_key: str,
    reranker_model_name: str,
    candidate_top_k: int,
    final_top_k: int,
    context_selection: str,
    num_queries: int,
    dense_agg: Dict[str, Any],
    reranked_agg: Dict[str, Any],
    final_agg: Dict[str, Any],
    timing: Dict[str, float],
) -> str:
    rows = [
        ("Document Hit@1", "document_hit_at_1"),
        ("Document Hit@5", "document_hit_at_5"),
        ("Document Recall@5", "document_recall_at_5"),
        ("Document Precision@5", "document_precision_at_5"),
        ("Document MRR", "document_mrr"),
        ("Chunk Hit@5", "chunk_hit_at_5"),
        ("Chunk Recall@5", "chunk_recall_at_5"),
        ("Chunk Precision@5", "chunk_precision_at_5"),
        ("Chunk MRR", "chunk_mrr"),
        ("Section Hit@5", "section_hit_at_5"),
        ("Section Precision@5", "section_precision_at_5"),
        ("Section MRR", "section_mrr"),
        ("Page Hit@1", "page_hit_at_1"),
        ("Page Hit@5", "page_hit_at_5"),
        ("Page Precision@5", "page_precision_at_5"),
        ("Soft Page Hit@5 (±1)", "soft_page_hit_at_5"),
        ("Label Hit@5", "label_hit_at_5"),
    ]

    def _delta_str(a: Optional[float], b: Optional[float]) -> str:
        if a is None or b is None:
            return "—"
        d = float(b) - float(a)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    target_p5 = final_agg.get("document_precision_at_5")
    target_r5 = final_agg.get("document_recall_at_5")
    cs_active = context_selection != "none"

    md: List[str] = []
    md.append("# Reranking + context selection report\n")
    md.append("## Конфигурация\n")
    md.append(f"- Embedding model: `{embedding_model_key}` (`{embedding_model_name}`)")
    md.append(f"- Reranker: `{reranker_key}` (`{reranker_model_name}`)")
    md.append(f"- context_selection: `{context_selection}`")
    md.append(f"- candidate_top_k = {candidate_top_k}, final_top_k = {final_top_k}")
    md.append(f"- queries = {num_queries}\n")

    md.append("## Цели проекта (по финальному пайплайну)\n")
    md.append(
        f"- Recall@5 ≥ 0.80 — "
        f"{'✓ достигнут' if target_r5 is not None and target_r5 >= 0.80 else '✗ не достигнут' if target_r5 is not None else '— нет данных'} "
        f"(final = {_fmt(target_r5)})"
    )
    md.append(
        f"- Precision@5 ≥ 0.70 — "
        f"{'✓ достигнут' if target_p5 is not None and target_p5 >= 0.70 else '✗ не достигнут' if target_p5 is not None else '— нет данных'} "
        f"(final = {_fmt(target_p5)})\n"
    )

    if cs_active:
        md.append("## Сравнение dense → reranked → final (с context selection)\n")
        md.append(
            "| Метрика | Dense top-5 | Reranked top-5 | Final top-5 | Δ (full) | Δ (CS only) |"
        )
        md.append("|---|---|---|---|---|---|")
        for label, key in rows:
            d = dense_agg.get(key)
            r = reranked_agg.get(key)
            f = final_agg.get(key)
            md.append(
                f"| {label} | {_fmt(d)} | {_fmt(r)} | {_fmt(f)} | "
                f"{_delta_str(d, f)} | {_delta_str(r, f)} |"
            )
    else:
        md.append("## Сравнение before / after reranking\n")
        md.append("| Метрика | Dense top-5 | Reranked top-5 | Δ |")
        md.append("|---|---|---|---|")
        for label, key in rows:
            d = dense_agg.get(key)
            r = reranked_agg.get(key)
            md.append(f"| {label} | {_fmt(d)} | {_fmt(r)} | {_delta_str(d, r)} |")
    md.append("")

    md.append("## Покрытие разметкой (сколько запросов имеют разметку для каждой метрики)\n")
    md.append("| Метрика | Coverage |")
    md.append("|---|---|")
    for label, key in rows:
        cov = final_agg.get(f"{key}_coverage", 0)
        md.append(f"| {label} | {cov} / {num_queries} |")
    md.append("")

    md.append("## Производительность\n")
    md.append("| Шаг | мс/запрос |")
    md.append("|---|---|")
    md.append(f"| query embedding   | {timing.get('avg_query_embedding_time_ms', 0):.2f} |")
    md.append(f"| dense retrieval    | {timing.get('avg_dense_retrieval_time_ms', 0):.2f} |")
    md.append(f"| reranking          | {timing.get('avg_reranking_time_ms', 0):.2f} |")
    md.append(f"| context selection  | {timing.get('avg_context_selection_time_ms', 0):.2f} |")
    md.append(f"| total              | {timing.get('avg_total_time_ms', 0):.2f} |")
    md.append("")

    md.append("## Рекомендации\n")
    if target_p5 is not None and target_p5 >= 0.70 and target_r5 is not None and target_r5 >= 0.80:
        md.append(
            "- Оба целевых порога достигнуты. Финальная связка — рабочая. "
            "Имеет смысл закрепить параметры пайплайна и зафиксировать reranker в RAG-сервисе."
        )
    else:
        if target_p5 is None or target_p5 < 0.70:
            if context_selection == "none":
                md.append(
                    "- Precision@5 ниже 0.70. Попробовать `--context-selection anchor_section` "
                    "или `anchor_document` — это формирование финального RAG-контекста "
                    "вокруг наиболее релевантного фрагмента."
                )
            else:
                md.append(
                    f"- Precision@5 (с context_selection={context_selection}) ниже 0.70. "
                    "Попробовать другой mode (`anchor_document`/`anchor_section`/`anchor_page`), "
                    "увеличить `candidate_top_k` или сменить reranker."
                )
        if target_r5 is None or target_r5 < 0.80:
            md.append(
                "- Recall@5 ниже 0.80. Ни reranker, ни context selection не могут вытащить "
                "документ, которого нет в dense top-N. Увеличьте `candidate_top_k` или "
                "смените embedding-модель."
            )

    if cs_active:
        delta_chunk_cs = (
            None if reranked_agg.get("chunk_hit_at_5") is None or final_agg.get("chunk_hit_at_5") is None
            else float(final_agg["chunk_hit_at_5"]) - float(reranked_agg["chunk_hit_at_5"])
        )
        if delta_chunk_cs is not None and delta_chunk_cs < -0.02:
            md.append(
                f"- ⚠️ Chunk Hit@5 просел на {delta_chunk_cs:+.4f} после context selection — "
                "anchor вытесняется добранными чанками. Стоит проверить `detailed_results.jsonl` "
                "или ослабить context_selection до `anchor_section`."
            )

    md.append(
        "- Сильное расхождение между document/section/page/chunk Hit@5 показывает, "
        "достаточно ли точно retrieval попадает на нужный фрагмент, а не только на нужный документ."
    )

    return "\n".join(md) + "\n"


def run_evaluation(
    *,
    queries_path: str | Path,
    embedding_model_key: str,
    embedding_config: str | Path,
    embeddings_dir: str | Path,
    reranker_key: str,
    reranker_config: str | Path,
    candidate_top_k: int = 30,
    final_top_k: int = 5,
    device: str = "auto",
    output_dir: str | Path,
    limit: Optional[int] = None,
    overwrite: bool = False,
    save_all_candidates: bool = False,
    text_preview_chars: int = 300,
    context_selection: str = "none",
    context_page_tolerance: int = 1,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    detailed_path = output_dir / "detailed_results.jsonl"
    if detailed_path.exists() and not overwrite:
        raise FileExistsError(
            f"{detailed_path} already exists. Pass --overwrite to replace."
        )
    ensure_dir(output_dir)

    if context_selection not in SUPPORTED_MODES:
        raise ValueError(
            f"Unknown context-selection mode: {context_selection!r}. "
            f"Supported: {SUPPORTED_MODES}"
        )

    embeddings, metadata, run_info = _load_embedding_artifacts(
        Path(embeddings_dir), embedding_model_key
    )
    chunk_id_to_index: Dict[str, int] = {}
    for i, m in enumerate(metadata):
        cid = m.get("id")
        if cid:
            chunk_id_to_index[str(cid)] = i

    logger.info(
        "Loaded %d embeddings (dim=%d) for %s from %s",
        embeddings.shape[0],
        embeddings.shape[1],
        embedding_model_key,
        embeddings_dir,
    )

    retriever = NumpyRetriever(
        embeddings, metadata, already_normalized=bool(run_info.get("normalize", True))
    )

    embedding_cfg = load_model_config(embedding_config, embedding_model_key)
    embedding_model = load_embedding_model(embedding_cfg, device=device)

    rer_cfg = load_reranker_config(reranker_config, reranker_key)
    reranker = build_reranker(rer_cfg, device=device)

    queries = read_jsonl_list(queries_path)
    if limit is not None and limit > 0:
        queries = queries[:limit]
    logger.info("Loaded %d queries from %s", len(queries), queries_path)

    detailed_records: List[Dict[str, Any]] = []
    dense_per_query: List[Dict[str, Any]] = []
    rerank_per_query: List[Dict[str, Any]] = []
    final_per_query: List[Dict[str, Any]] = []

    times_qe: List[float] = []
    times_dr: List[float] = []
    times_rr: List[float] = []
    times_cs: List[float] = []
    times_total: List[float] = []

    for q in tqdm(queries, desc=f"rerank:{embedding_model_key}+{reranker_key}", file=sys.stdout, dynamic_ncols=True):
        query_id = q.get("query_id")
        query_text = q.get("query") or ""

        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        q_vec = embedding_model.encode_queries([query_text], show_progress_bar=False)[0]
        time_qe = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        candidates = retriever.search(q_vec, top_k=candidate_top_k)
        time_dr = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        reranked, rer_scores, dense_ranks = _rerank(
            candidates,
            query_text=query_text,
            metadata=metadata,
            chunk_id_to_index=chunk_id_to_index,
            reranker=reranker,
        )
        time_rr = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        final_hits, final_scores, final_dranks = select_context_top_k(
            reranked=reranked,
            reranker_scores=rer_scores,
            dense_ranks_in_rerank=dense_ranks,
            metadata=metadata,
            mode=context_selection,
            final_top_k=final_top_k,
            page_tolerance=context_page_tolerance,
        )
        time_cs = (time.perf_counter() - t3) * 1000.0

        time_total = (time.perf_counter() - t_total_start) * 1000.0

        times_qe.append(time_qe)
        times_dr.append(time_dr)
        times_rr.append(time_rr)
        times_cs.append(time_cs)
        times_total.append(time_total)

        dense_topk = candidates[:final_top_k]
        reranked_topk = reranked[:final_top_k]

        dense_metrics = evaluate_query_advanced(
            dense_topk, q, final_top_k=final_top_k
        )
        reranked_metrics = evaluate_query_advanced(
            reranked_topk, q, final_top_k=final_top_k
        )
        final_metrics = evaluate_query_advanced(
            final_hits, q, final_top_k=final_top_k
        )
        dense_per_query.append(dense_metrics)
        rerank_per_query.append(reranked_metrics)
        final_per_query.append(final_metrics)

        if save_all_candidates:
            dense_dump = candidates
            reranked_dump = list(zip(reranked, rer_scores, dense_ranks))
        else:
            dense_dump = dense_topk
            reranked_dump = list(zip(reranked_topk, rer_scores[:final_top_k], dense_ranks[:final_top_k]))
        final_dump = list(zip(final_hits, final_scores, final_dranks))

        record = {
            "query_id": query_id,
            "query": query_text,
            "difficulty": q.get("difficulty"),
            "query_type": q.get("query_type"),
            "embedding_model_key": embedding_model_key,
            "reranker_key": reranker_key,
            "context_selection": context_selection,
            "expected_document_ids": q.get("expected_document_ids") or [],
            "expected_chunk_ids": q.get("expected_chunk_ids") or [],
            "expected_section_ids": q.get("expected_section_ids") or [],
            "expected_section_titles": q.get("expected_section_titles") or [],
            "expected_section_keywords": q.get("expected_section_keywords") or [],
            "expected_labels": q.get("expected_labels") or [],
            "source_evidence": q.get("source_evidence") or [],
            "dense_top_results": [
                _hit_to_dense_dict(h, text_preview_chars=text_preview_chars)
                for h in dense_dump
            ],
            "reranked_top_results": [
                _hit_to_reranked_dict(
                    h,
                    new_rank=h.rank,
                    dense_rank=d_rank,
                    reranker_score=score,
                    text_preview_chars=text_preview_chars,
                )
                for h, score, d_rank in reranked_dump
            ],
            "final_top_results": [
                _hit_to_reranked_dict(
                    h,
                    new_rank=h.rank,
                    dense_rank=d_rank,
                    reranker_score=score,
                    text_preview_chars=text_preview_chars,
                )
                for h, score, d_rank in final_dump
            ],
            "dense_metrics": dense_metrics,
            "reranked_metrics": reranked_metrics,
            "final_metrics": final_metrics,
            "timings_ms": {
                "query_embedding": round(time_qe, 3),
                "dense_retrieval": round(time_dr, 3),
                "reranking": round(time_rr, 3),
                "context_selection": round(time_cs, 3),
                "total": round(time_total, 3),
            },
        }
        detailed_records.append(record)

    write_jsonl(detailed_path, detailed_records)
    logger.info("Wrote detailed results: %s", detailed_path)

    dense_agg = aggregate_advanced(dense_per_query)
    reranked_agg = aggregate_advanced(rerank_per_query)
    final_agg = aggregate_advanced(final_per_query)

    timing = {
        "avg_query_embedding_time_ms": round(_avg(times_qe), 3),
        "avg_dense_retrieval_time_ms": round(_avg(times_dr), 3),
        "avg_reranking_time_ms": round(_avg(times_rr), 3),
        "avg_context_selection_time_ms": round(_avg(times_cs), 3),
        "avg_total_time_ms": round(_avg(times_total), 3),
    }

    csv_row = _build_csv_row(
        embedding_model_key=embedding_model_key,
        reranker_key=reranker_key,
        context_selection=context_selection,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
        num_queries=len(queries),
        dense=dense_agg,
        reranked=reranked_agg,
        final=final_agg,
        timing=timing,
    )
    csv_path = output_dir / "rerank_comparison_metrics.csv"
    json_path = output_dir / "rerank_comparison_metrics.json"
    _write_csv(csv_row, csv_path)
    write_json(
        json_path,
        {
            "config": {
                "embedding_model_key": embedding_model_key,
                "embedding_model_name": run_info.get("model_name"),
                "reranker_key": reranker_key,
                "reranker_model_name": rer_cfg.get("model_name"),
                "candidate_top_k": candidate_top_k,
                "final_top_k": final_top_k,
                "context_selection": context_selection,
                "context_page_tolerance": context_page_tolerance,
                "device": device,
                "queries_path": str(queries_path),
                "embeddings_dir": str(embeddings_dir),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "dense_metrics": dense_agg,
            "reranked_metrics": reranked_agg,
            "final_metrics": final_agg,
            "timing": timing,
            "row": csv_row,
        },
    )
    logger.info("Wrote aggregate metrics: %s, %s", csv_path, json_path)

    md_text = _build_markdown_report(
        embedding_model_key=embedding_model_key,
        embedding_model_name=str(run_info.get("model_name") or ""),
        reranker_key=reranker_key,
        reranker_model_name=str(rer_cfg.get("model_name") or ""),
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
        context_selection=context_selection,
        num_queries=len(queries),
        dense_agg=dense_agg,
        reranked_agg=reranked_agg,
        final_agg=final_agg,
        timing=timing,
    )
    md_path = output_dir / "rerank_report.md"
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote markdown report: %s", md_path)

    return {
        "num_queries": len(queries),
        "context_selection": context_selection,
        "dense_metrics": dense_agg,
        "reranked_metrics": reranked_agg,
        "final_metrics": final_agg,
        "timing": timing,
        "paths": {
            "detailed_results": str(detailed_path),
            "comparison_csv": str(csv_path),
            "comparison_json": str(json_path),
            "report_md": str(md_path),
        },
    }


def _avg(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

CSV_COLUMNS = [
    "embedding_model_key",
    "reranker_key",
    "context_selection",
    "number_of_queries",
    "candidate_top_k",
    "final_top_k",
    "dense_document_hit_at_1",
    "dense_document_hit_at_5",
    "dense_document_recall_at_5",
    "dense_document_precision_at_5",
    "dense_document_mrr",
    "dense_chunk_hit_at_5",
    "dense_chunk_precision_at_5",
    "dense_section_hit_at_5",
    "dense_page_hit_at_5",
    "dense_soft_page_hit_at_5",
    "reranked_document_hit_at_1",
    "reranked_document_hit_at_5",
    "reranked_document_recall_at_5",
    "reranked_document_precision_at_5",
    "reranked_document_mrr",
    "reranked_chunk_hit_at_5",
    "reranked_chunk_precision_at_5",
    "reranked_section_hit_at_5",
    "reranked_page_hit_at_5",
    "reranked_soft_page_hit_at_5",
    "final_document_hit_at_1",
    "final_document_hit_at_5",
    "final_document_recall_at_5",
    "final_document_precision_at_5",
    "final_document_mrr",
    "final_chunk_hit_at_5",
    "final_chunk_precision_at_5",
    "final_section_hit_at_5",
    "final_page_hit_at_5",
    "final_soft_page_hit_at_5",
    "delta_document_precision_at_5",
    "delta_document_recall_at_5",
    "delta_chunk_hit_at_5",
    "delta_section_hit_at_5",
    "delta_page_hit_at_5",
    "delta_soft_page_hit_at_5",
    "delta_cs_document_precision_at_5",
    "delta_cs_chunk_hit_at_5",
    "delta_cs_section_hit_at_5",
    "delta_cs_page_hit_at_5",
    "avg_query_embedding_time_ms",
    "avg_dense_retrieval_time_ms",
    "avg_reranking_time_ms",
    "avg_context_selection_time_ms",
    "avg_total_time_ms",
]


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 4)


def _build_csv_row(
    *,
    embedding_model_key: str,
    reranker_key: str,
    context_selection: str,
    candidate_top_k: int,
    final_top_k: int,
    num_queries: int,
    dense: Dict[str, Any],
    reranked: Dict[str, Any],
    final: Dict[str, Any],
    timing: Dict[str, float],
) -> Dict[str, Any]:
    def g(d: Dict[str, Any], k: str) -> Any:
        return d.get(k)

    row: Dict[str, Any] = {
        "embedding_model_key": embedding_model_key,
        "reranker_key": reranker_key,
        "context_selection": context_selection,
        "number_of_queries": num_queries,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
    }

    metric_keys = [
        "document_hit_at_1",
        "document_hit_at_5",
        "document_recall_at_5",
        "document_precision_at_5",
        "document_mrr",
        "chunk_hit_at_5",
        "chunk_precision_at_5",
        "section_hit_at_5",
        "page_hit_at_5",
        "soft_page_hit_at_5",
    ]
    for k in metric_keys:
        row[f"dense_{k}"] = g(dense, k)
        row[f"reranked_{k}"] = g(reranked, k)
        row[f"final_{k}"] = g(final, k)

    for k in [
        "document_precision_at_5",
        "document_recall_at_5",
        "chunk_hit_at_5",
        "section_hit_at_5",
        "page_hit_at_5",
        "soft_page_hit_at_5",
    ]:
        row[f"delta_{k}"] = _delta(dense.get(k), final.get(k))

    for k in [
        "document_precision_at_5",
        "chunk_hit_at_5",
        "section_hit_at_5",
        "page_hit_at_5",
    ]:
        row[f"delta_cs_{k}"] = _delta(reranked.get(k), final.get(k))

    row.update(timing)
    return row


def _write_csv(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})



def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense retrieval → reranking → before/after metrics"
    )
    parser.add_argument("--queries-path", required=True, type=str)
    parser.add_argument("--embedding-model-key", required=True, type=str)
    parser.add_argument("--embedding-config", required=True, type=str)
    parser.add_argument("--embeddings-dir", required=True, type=str)
    parser.add_argument("--reranker-key", required=True, type=str)
    parser.add_argument("--reranker-config", required=True, type=str)
    parser.add_argument("--candidate-top-k", type=int, default=30)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument(
        "--context-selection",
        type=str,
        default="none",
        choices=list(SUPPORTED_MODES),
        help=(
            "Anchor-based context expansion mode. "
            "'none' = top-K reranker'a без изменений; "
            "'anchor_document' = добираем чанки того же документа; "
            "'anchor_section' = только из того же раздела; "
            "'anchor_page' = только из чанков с пересекающимися страницами."
        ),
    )
    parser.add_argument(
        "--context-page-tolerance",
        type=int,
        default=1,
        help="Допуск ±N страниц при матчинге same-page (используется в anchor_page).",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-all-candidates",
        action="store_true",
        help="Save full candidate_top_k in detailed_results (not just final_top_k).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        verbose=args.verbose,
        log_file=out_dir / "evaluate_with_reranker.log",
        name="evaluate_with_reranker",
    )
    logger.info("Args: %s", vars(args))
    try:
        run_evaluation(
            queries_path=args.queries_path,
            embedding_model_key=args.embedding_model_key,
            embedding_config=args.embedding_config,
            embeddings_dir=args.embeddings_dir,
            reranker_key=args.reranker_key,
            reranker_config=args.reranker_config,
            candidate_top_k=args.candidate_top_k,
            final_top_k=args.final_top_k,
            device=args.device,
            output_dir=args.output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            save_all_candidates=args.save_all_candidates,
            context_selection=args.context_selection,
            context_page_tolerance=args.context_page_tolerance,
        )
    except FileExistsError as e:
        logger.error("%s", e)
        return 2
    except Exception:
        logger.exception("Reranking evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
