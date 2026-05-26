from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..io_utils import read_yaml
from .retrieval_service import RetrievalService
from .schemas import (
    BatchSearchRequest,
    BatchSearchResponse,
    BatchTiming,
    DocumentsResponse,
    HealthResponse,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)

_service: Optional[RetrievalService] = None
_api_config: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service, _api_config
    config_path = os.environ.get("RAG_API_CONFIG", "configs/api.yaml")
    logger.info("Loading API config from: %s", config_path)

    _api_config = read_yaml(config_path)
    _setup_file_logging(_api_config)

    _service = RetrievalService(_api_config)
    try:
        _service.load()
    except Exception:
        logger.exception("Failed to load RetrievalService")
        raise

    app.state.service = _service
    app.state.config = _api_config
    yield
    logger.info("Shutting down RAG Retrieval API")


def _setup_file_logging(cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("log_requests", True):
        return
    log_dir = log_cfg.get("log_dir", "logs/api")
    log_file = log_cfg.get("log_file", "rag_api.log")
    import os as _os
    _os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(f"{log_dir}/{log_file}", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

def _create_app(cfg: Optional[Dict[str, Any]] = None) -> FastAPI:
    api_meta = (cfg or {}).get("api", {})
    app = FastAPI(
        title=api_meta.get("title", "Medical RAG Retrieval API"),
        version=api_meta.get("version", "0.1.0"),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = _create_app()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f trace_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        trace_id,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


def _get_service() -> RetrievalService:
    if _service is None or not _service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Check /ready for details.",
        )
    return _service


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    svc = _service
    if svc is None:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "service": "rag-retrieval-api", "loaded": False},
        )
    return svc.get_health()


@app.get("/ready", response_model=ReadyResponse, tags=["health"])
def ready():
    svc = _service
    if svc is None:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "checks": {
                    "embeddings_loaded": False,
                    "metadata_loaded": False,
                    "embedding_model_loaded": False,
                    "reranker_loaded": False,
                },
            },
        )
    result = svc.get_ready()
    status_code = 200 if result["ready"] else 503
    return JSONResponse(status_code=status_code, content=result)


@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(request: SearchRequest):
    svc = _get_service()
    cfg = _api_config
    perf = cfg.get("performance", {})
    max_q_len = perf.get("max_query_length_chars", 4000)

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty after stripping")
    if len(query) > max_q_len:
        raise HTTPException(
            status_code=400,
            detail=f"Query exceeds max length of {max_q_len} chars",
        )

    filters_dict: Optional[Dict[str, Any]] = None
    if request.filters:
        f = request.filters
        filters_dict = {
            "document_id": f.document_id,
            "label": f.label,
            "specialty": f.specialty,
        }

    try:
        result = svc.search(
            query=query,
            candidate_top_k=request.candidate_top_k,
            final_top_k=request.final_top_k,
            use_reranker=request.use_reranker,
            context_selection=request.context_selection,
            filters=filters_dict,
            include_text=request.include_text,
            include_embedding_text=request.include_embedding_text,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Search failed for query=%r", query)
        raise HTTPException(status_code=500, detail=f"Internal search error: {e}")

    return result


@app.post("/batch_search", response_model=BatchSearchResponse, tags=["retrieval"])
def batch_search(request: BatchSearchRequest):
    svc = _get_service()
    cfg = _api_config
    perf = cfg.get("performance", {})
    max_batch = perf.get("max_batch_size", 20)
    max_q_len = perf.get("max_query_length_chars", 4000)

    if len(request.queries) > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.queries)} exceeds limit {max_batch}",
        )

    queries = [q.strip() for q in request.queries]
    for i, q in enumerate(queries):
        if not q:
            raise HTTPException(status_code=400, detail=f"Query at index {i} is empty")
        if len(q) > max_q_len:
            raise HTTPException(
                status_code=400,
                detail=f"Query at index {i} exceeds max length {max_q_len}",
            )

    filters_dict: Optional[Dict[str, Any]] = None
    if request.filters:
        f = request.filters
        filters_dict = {
            "document_id": f.document_id,
            "label": f.label,
            "specialty": f.specialty,
        }

    try:
        items, total_ms, avg_ms = svc.batch_search(
            queries=queries,
            candidate_top_k=request.candidate_top_k,
            final_top_k=request.final_top_k,
            use_reranker=request.use_reranker,
            context_selection=request.context_selection,
            filters=filters_dict,
            include_text=request.include_text,
            include_embedding_text=request.include_embedding_text,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Batch search failed")
        raise HTTPException(status_code=500, detail=f"Internal batch search error: {e}")

    return {
        "items": items,
        "batch_timing": {"total_time_ms": total_ms, "avg_time_ms": avg_ms},
    }


@app.get("/config", tags=["info"])
def get_config():
    return _api_config


@app.get("/documents", response_model=DocumentsResponse, tags=["info"])
def documents():
    svc = _get_service()
    docs = svc.documents()
    return {"documents": docs, "total_documents": len(docs)}


@app.get("/stats", response_model=StatsResponse, tags=["info"])
def stats():
    svc = _get_service()
    return svc.stats()


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Medical RAG Retrieval API")
    parser.add_argument(
        "--config",
        default="configs/api.yaml",
        help="Path to api.yaml config (default: configs/api.yaml)",
    )
    parser.add_argument("--host", default=None, help="Override host from config")
    parser.add_argument("--port", default=None, type=int, help="Override port from config")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)")
    args = parser.parse_args()

    os.environ["RAG_API_CONFIG"] = args.config

    try:
        cfg = read_yaml(args.config)
    except FileNotFoundError:
        print(f"Config not found: {args.config}")
        raise SystemExit(1)

    api_cfg = cfg.get("api", {})
    log_level = api_cfg.get("log_level", "info")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    host = args.host or api_cfg.get("host", "0.0.0.0")
    port = args.port or api_cfg.get("port", 8000)

    logger.info("Starting RAG Retrieval API on %s:%d", host, port)
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
