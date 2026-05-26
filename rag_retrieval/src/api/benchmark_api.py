from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _call_search(
    url: str,
    query: str,
    candidate_top_k: int,
    final_top_k: int,
    use_reranker: bool,
    context_selection: str,
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "use_reranker": use_reranker,
        "context_selection": context_selection,
        "include_text": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            data = resp.json()
            server_ms = data.get("timing", {}).get("total_time_ms")
            return {
                "success": True,
                "status_code": resp.status_code,
                "client_latency_ms": round(elapsed_ms, 2),
                "server_latency_ms": server_ms,
                "num_results": len(data.get("results", [])),
                "trace_id": data.get("trace_id"),
                "error": None,
            }
        else:
            return {
                "success": False,
                "status_code": resp.status_code,
                "client_latency_ms": round(elapsed_ms, 2),
                "server_latency_ms": None,
                "num_results": 0,
                "trace_id": None,
                "error": resp.text[:200],
            }
    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "status_code": None,
            "client_latency_ms": round(elapsed_ms, 2),
            "server_latency_ms": None,
            "num_results": 0,
            "trace_id": None,
            "error": "timeout",
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "status_code": None,
            "client_latency_ms": round(elapsed_ms, 2),
            "server_latency_ms": None,
            "num_results": 0,
            "trace_id": None,
            "error": str(e),
        }


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_v):
        return float(sorted_v[-1])
    frac = idx - lo
    return float(sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac)


def run_benchmark(
    url: str,
    queries_path: str,
    query_field: str,
    limit: Optional[int],
    output: str,
    candidate_top_k: int = 30,
    final_top_k: int = 5,
    use_reranker: bool = True,
    context_selection: str = "anchor_page",
    concurrency: int = 1,
    timeout: float = 30.0,
    target_latency_ms: float = 2000.0,
) -> Dict[str, Any]:
    records = _read_jsonl(queries_path)
    if limit and limit > 0:
        records = records[:limit]
    logger.info("Loaded %d queries from %s", len(records), queries_path)

    queries = []
    for r in records:
        q = r.get(query_field) or ""
        q = q.strip()
        if q:
            queries.append(q)
    logger.info("Will benchmark %d non-empty queries", len(queries))

    if not queries:
        raise ValueError(f"No queries found in {queries_path} under field '{query_field}'")

    call_args = dict(
        url=url,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
        use_reranker=use_reranker,
        context_selection=context_selection,
        timeout=timeout,
    )

    individual_results: List[Dict[str, Any]] = []
    t_bench_start = time.perf_counter()

    if concurrency <= 1:
        for i, q in enumerate(queries, start=1):
            logger.info("[%d/%d] query: %s...", i, len(queries), q[:60])
            r = _call_search(query=q, **call_args)
            r["query"] = q
            individual_results.append(r)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_call_search, query=q, **call_args): q for q in queries}
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                q = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        "success": False,
                        "status_code": None,
                        "client_latency_ms": 0,
                        "server_latency_ms": None,
                        "num_results": 0,
                        "trace_id": None,
                        "error": str(e),
                    }
                r["query"] = q
                individual_results.append(r)
                logger.info("[%d/%d] done", i, len(queries))

    bench_elapsed_ms = (time.perf_counter() - t_bench_start) * 1000.0

    successes = [r for r in individual_results if r["success"]]
    client_latencies = [r["client_latency_ms"] for r in successes]
    server_latencies = [r["server_latency_ms"] for r in successes if r["server_latency_ms"] is not None]
    errors = [r for r in individual_results if not r["success"]]

    use_latencies = server_latencies if server_latencies else client_latencies

    metrics = {
        "num_requests": len(individual_results),
        "num_success": len(successes),
        "num_errors": len(errors),
        "success_rate": round(len(successes) / len(individual_results), 4) if individual_results else 0.0,
        "avg_latency_ms": round(sum(use_latencies) / len(use_latencies), 2) if use_latencies else None,
        "p50_latency_ms": round(_percentile(use_latencies, 50), 2) if use_latencies else None,
        "p95_latency_ms": round(_percentile(use_latencies, 95), 2) if use_latencies else None,
        "max_latency_ms": round(max(use_latencies), 2) if use_latencies else None,
        "min_latency_ms": round(min(use_latencies), 2) if use_latencies else None,
        "target_latency_ms": target_latency_ms,
        "target_pass_rate": (
            round(
                sum(1 for l in use_latencies if l < target_latency_ms) / len(use_latencies), 4
            )
            if use_latencies else None
        ),
        "bench_wall_time_ms": round(bench_elapsed_ms, 2),
        "latency_source": "server" if server_latencies else "client",
        "config": {
            "url": url,
            "queries_path": queries_path,
            "query_field": query_field,
            "limit": limit,
            "candidate_top_k": candidate_top_k,
            "final_top_k": final_top_k,
            "use_reranker": use_reranker,
            "context_selection": context_selection,
            "concurrency": concurrency,
        },
        "errors": [
            {"query": e.get("query", "")[:80], "error": e.get("error"), "status": e.get("status_code")}
            for e in errors[:10]
        ],
        "individual_results": individual_results,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON results: %s", output_path)

    csv_path = output_path.with_suffix(".csv")
    csv_cols = [
        "query", "success", "status_code",
        "client_latency_ms", "server_latency_ms", "num_results", "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in individual_results:
            writer.writerow({k: r.get(k, "") for k in csv_cols})
    logger.info("Saved CSV results: %s", csv_path)

    md_path = output_path.with_name("api_benchmark_report.md")
    _write_md_report(md_path, metrics)
    logger.info("Saved Markdown report: %s", md_path)

    return metrics


def _write_md_report(path: Path, m: Dict[str, Any]) -> None:
    avg = m["avg_latency_ms"]
    p50 = m["p50_latency_ms"]
    p95 = m["p95_latency_ms"]
    mx = m["max_latency_ms"]
    target = m["target_latency_ms"]
    tpr = m["target_pass_rate"]
    src = m.get("latency_source", "client")
    cfg = m.get("config", {})

    lines = [
        "# API Benchmark Report\n",
        "## Конфигурация\n",
        f"- URL: `{cfg.get('url')}`",
        f"- Запросы: `{cfg.get('queries_path')}` (поле: `{cfg.get('query_field')}`, limit: {cfg.get('limit')})",
        f"- candidate_top_k: {cfg.get('candidate_top_k')}, final_top_k: {cfg.get('final_top_k')}",
        f"- use_reranker: {cfg.get('use_reranker')}, context_selection: {cfg.get('context_selection')}",
        f"- concurrency: {cfg.get('concurrency')}\n",
        "## Результаты\n",
        f"| Метрика | Значение |",
        f"|---|---|",
        f"| Запросов всего | {m['num_requests']} |",
        f"| Успешных | {m['num_success']} |",
        f"| Ошибок | {m['num_errors']} |",
        f"| Success rate | {m['success_rate']:.1%} |",
        f"| Avg latency ({src}) | {avg:.1f} мс |" if avg is not None else "| Avg latency | — |",
        f"| P50 latency ({src}) | {p50:.1f} мс |" if p50 is not None else "| P50 latency | — |",
        f"| P95 latency ({src}) | {p95:.1f} мс |" if p95 is not None else "| P95 latency | — |",
        f"| Max latency ({src}) | {mx:.1f} мс |" if mx is not None else "| Max latency | — |",
        f"| Target latency | {target:.0f} мс |",
        f"| Target pass rate | {tpr:.1%} |" if tpr is not None else "| Target pass rate | — |",
        "",
        "## Вывод\n",
    ]
    if tpr is not None and tpr >= 1.0:
        lines.append(
            f"✓ **Все запросы уложились в целевое время {target:.0f} мс** "
            f"(pass rate = {tpr:.1%})."
        )
    elif tpr is not None and tpr >= 0.95:
        lines.append(
            f"✓ **95% запросов уложились в целевое время {target:.0f} мс** "
            f"(pass rate = {tpr:.1%})."
        )
    elif tpr is not None:
        lines.append(
            f"⚠ Только {tpr:.1%} запросов уложились в целевое время {target:.0f} мс. "
            "Возможно, нужно оптимизировать конфигурацию или уменьшить candidate_top_k."
        )

    if m["num_errors"] > 0:
        lines.append(f"\n⚠ Было {m['num_errors']} ошибочных запросов.")
        lines.append("\n### Примеры ошибок\n")
        for e in m.get("errors", [])[:5]:
            lines.append(f"- `{e.get('query', '')}` — {e.get('error')} (status={e.get('status')})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark /search API endpoint")
    p.add_argument("--url", default="http://localhost:8000/search")
    p.add_argument("--queries-path", required=True)
    p.add_argument("--query-field", default="patient_case")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="runs/api_benchmark/api_benchmark_results.json")
    p.add_argument("--candidate-top-k", type=int, default=30)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument(
        "--context-selection",
        default="anchor_page",
        choices=["none", "anchor_page", "anchor_section", "anchor_document"],
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--target-latency-ms", type=float, default=2000.0)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    metrics = run_benchmark(
        url=args.url,
        queries_path=args.queries_path,
        query_field=args.query_field,
        limit=args.limit,
        output=args.output,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
        use_reranker=not args.no_reranker,
        context_selection=args.context_selection,
        concurrency=args.concurrency,
        timeout=args.timeout,
        target_latency_ms=args.target_latency_ms,
    )
    print("\n=== Benchmark Summary ===")
    print(f"  Requests:     {metrics['num_requests']}")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    avg = metrics["avg_latency_ms"]
    p95 = metrics["p95_latency_ms"]
    mx = metrics["max_latency_ms"]
    src = metrics.get("latency_source", "client")
    print(f"  Avg latency ({src}): {avg:.1f} ms" if avg is not None else "  Avg latency: N/A")
    print(f"  P95 latency ({src}): {p95:.1f} ms" if p95 is not None else "  P95 latency: N/A")
    print(f"  Max latency ({src}): {mx:.1f} ms" if mx is not None else "  Max latency: N/A")
    tpr = metrics["target_pass_rate"]
    target = metrics["target_latency_ms"]
    print(f"  Target ({target:.0f}ms) pass rate: {tpr:.1%}" if tpr is not None else "  Target pass rate: N/A")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
