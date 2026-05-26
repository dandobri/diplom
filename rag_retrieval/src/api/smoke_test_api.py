from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import requests


def _check(condition: bool, name: str, detail: str = "") -> bool:
    if condition:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}" + (f": {detail}" if detail else ""))
    return condition


def _get(url: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(url, timeout=timeout)
        return resp
    except Exception as e:
        print(f"  ERROR: GET {url} — {e}")
        return None


def _post(url: str, payload: Dict[str, Any], timeout: float = 30.0):
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        return resp
    except Exception as e:
        print(f"  ERROR: POST {url} — {e}")
        return None


def run_smoke_test(base_url: str, query: str, target_latency_ms: float = 2000.0) -> bool:
    base_url = base_url.rstrip("/")
    all_ok = True

    print("\n[1/6] GET /health")
    resp = _get(f"{base_url}/health")
    if resp is None:
        print("  ✗ Could not connect to server. Is it running?")
        return False
    ok = _check(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
    all_ok &= ok
    if ok:
        data = resp.json()
        _check(data.get("status") == "ok", "status == 'ok'", str(data.get("status")))
        _check(data.get("loaded") is True, "loaded == true")
        num_chunks = data.get("num_chunks", 0)
        _check(num_chunks > 0, f"num_chunks > 0 (got {num_chunks})")
        print(f"    device={data.get('device')}, uptime={data.get('uptime_sec')}s, chunks={num_chunks}")

    print("\n[2/6] GET /ready")
    resp = _get(f"{base_url}/ready")
    if resp is not None:
        _check(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        if resp.status_code == 200:
            data = resp.json()
            _check(data.get("ready") is True, "ready == true")
            checks = data.get("checks", {})
            for k, v in checks.items():
                _check(v, f"checks.{k}")

    print("\n[3/6] POST /search (full pipeline)")
    payload = {
        "query": query,
        "candidate_top_k": 30,
        "final_top_k": 5,
        "use_reranker": True,
        "context_selection": "anchor_page",
        "include_text": True,
    }
    resp = _post(f"{base_url}/search", payload)
    if resp is not None and _check(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})"):
        data = resp.json()
        results: List[Dict] = data.get("results", [])
        _check(len(results) >= 1, f"At least 1 result (got {len(results)})")

        if results:
            r0 = results[0]
            _check("chunk_id" in r0 and r0["chunk_id"], "result[0].chunk_id present")
            _check("document_id" in r0, "result[0].document_id present")
            _check("page_start" in r0, "result[0].page_start present")
            _check("page_end" in r0, "result[0].page_end present")
            _check("scores" in r0 and "dense_score" in r0.get("scores", {}), "result[0].scores.dense_score present")
            _check("context_source" in r0, "result[0].context_source present")

        timing = data.get("timing", {})
        total_ms = timing.get("total_time_ms")
        _check(total_ms is not None, "timing.total_time_ms present")
        if total_ms is not None:
            under_target = total_ms < target_latency_ms
            _check(
                under_target,
                f"total_time_ms {total_ms:.1f}ms < {target_latency_ms:.0f}ms",
                f"SLOW: {total_ms:.1f}ms",
            )
            all_ok &= under_target
            print(f"    Timing breakdown:")
            for k, v in timing.items():
                print(f"      {k}: {v:.1f}ms")

        _check("trace_id" in data and data["trace_id"], "trace_id present")
    else:
        all_ok = False

    print("\n[4/6] POST /search (no reranker)")
    payload_fast = {
        "query": query,
        "candidate_top_k": 10,
        "final_top_k": 3,
        "use_reranker": False,
        "context_selection": "none",
        "include_text": False,
    }
    resp = _post(f"{base_url}/search", payload_fast)
    if resp is not None:
        _check(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        if resp.status_code == 200:
            data = resp.json()
            _check(len(data.get("results", [])) >= 1, "At least 1 result")
            ms = data.get("timing", {}).get("total_time_ms")
            if ms is not None:
                print(f"    No-reranker total: {ms:.1f}ms")

    print("\n[5/6] GET /stats")
    resp = _get(f"{base_url}/stats")
    if resp is not None:
        _check(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        if resp.status_code == 200:
            data = resp.json()
            _check(data.get("num_chunks", 0) > 0, f"num_chunks > 0 (got {data.get('num_chunks')})")
            _check(data.get("num_documents", 0) > 0, f"num_documents > 0 (got {data.get('num_documents')})")

    print("\n[6/6] GET /documents")
    resp = _get(f"{base_url}/documents")
    if resp is not None:
        _check(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        if resp.status_code == 200:
            data = resp.json()
            docs = data.get("documents", [])
            _check(len(docs) > 0, f"At least 1 document (got {len(docs)})")
            if docs:
                d0 = docs[0]
                _check("document_id" in d0, "document_id present in first doc")
                _check("num_chunks" in d0, "num_chunks present in first doc")

    print()
    if all_ok:
        print("✓ ALL SMOKE TESTS PASSED")
    else:
        print("✗ SOME SMOKE TESTS FAILED")
    return all_ok


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Smoke test for RAG Retrieval API")
    p.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    p.add_argument(
        "--query",
        default="Пациент 58 лет, давящая боль за грудиной при нагрузке",
        help="Test query",
    )
    p.add_argument("--target-latency-ms", type=float, default=2000.0)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    print(f"Smoke testing RAG Retrieval API at {args.url}")
    print(f"Query: {args.query!r}")
    ok = run_smoke_test(
        base_url=args.url,
        query=args.query,
        target_latency_ms=args.target_latency_ms,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
