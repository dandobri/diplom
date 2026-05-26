from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sys
from tqdm import tqdm

from .embedding_models import (
    EmbeddingModel,
    ModelConfig,
    load_embedding_model,
    resolve_device,
)
from .io_utils import (
    ensure_dir,
    load_chunks_any,
    load_model_config,
    setup_logging,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)


METADATA_DROP_FIELDS = ()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> None:
    """Защита от перезаписи готового артефакта."""
    emb_path = output_dir / "embeddings.npy"
    if emb_path.exists():
        if overwrite:
            logger.warning("--overwrite is set, existing %s will be replaced", emb_path)
        elif resume:
            logger.info("--resume is set, existing %s will be reused if compatible", emb_path)
        else:
            raise FileExistsError(
                f"{emb_path} already exists. Pass --overwrite or --resume."
            )


def _build_run_info(
    model: EmbeddingModel,
    *,
    number_of_chunks: int,
    embedding_dim: int,
    batch_size: int,
    device: str,
    cuda_available: bool,
    gpu_name: Optional[str],
    started_at: str,
    finished_at: str,
    total_encoding_time_sec: float,
) -> Dict[str, Any]:
    cfg = model.config
    avg_per_chunk = (
        total_encoding_time_sec / number_of_chunks if number_of_chunks else 0.0
    )
    return {
        "model_key": cfg.model_key,
        "model_name": cfg.model_name,
        "backend": cfg.backend,
        "embedding_dim": int(embedding_dim),
        "number_of_chunks": int(number_of_chunks),
        "batch_size": int(batch_size),
        "max_length": int(cfg.max_length),
        "normalize": bool(cfg.normalize),
        "document_prefix": cfg.document_prefix,
        "query_prefix": cfg.query_prefix,
        "query_instruction": cfg.query_instruction,
        "trust_remote_code": bool(cfg.trust_remote_code),
        "device": device,
        "cuda_available": bool(cuda_available),
        "gpu_name": gpu_name,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "started_at": started_at,
        "finished_at": finished_at,
        "created_at": _utc_now_iso(),
        "total_encoding_time_sec": round(float(total_encoding_time_sec), 4),
        "avg_encoding_time_per_chunk_sec": round(float(avg_per_chunk), 6),
    }


def _validate_chunks_for_embedding(chunks: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Отсеивает чанки, у которых нет embedding_text.

    Возвращает (valid_chunks, errors).
    """
    valid: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        emb_text = ch.get("embedding_text")
        if not isinstance(emb_text, str) or not emb_text.strip():
            errors.append(
                {
                    "index": idx,
                    "id": ch.get("id"),
                    "reason": "missing or empty embedding_text",
                }
            )
            continue
        if not ch.get("id"):
            errors.append(
                {
                    "index": idx,
                    "id": None,
                    "reason": "missing id",
                }
            )
            continue
        valid.append(ch)
    return valid, errors


def build_embeddings_for_model(
    *,
    model_key: str,
    chunks_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = False,
    trust_remote_code: Optional[bool] = None,
) -> Dict[str, Any]:
    """Программный entry point. Возвращает run_info."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    _check_output_dir(output_dir, overwrite=overwrite, resume=resume)

    model_cfg = load_model_config(config_path, model_key)
    if trust_remote_code is not None:
        model_cfg["trust_remote_code"] = bool(trust_remote_code)
    if batch_size is not None:
        model_cfg["batch_size"] = int(batch_size)

    chunks = load_chunks_any(chunks_path)
    logger.info("Loaded %d raw chunks from %s", len(chunks), chunks_path)
    if limit is not None and limit > 0:
        chunks = chunks[:limit]
        logger.info("--limit=%d applied, processing %d chunks", limit, len(chunks))

    valid_chunks, error_records = _validate_chunks_for_embedding(chunks)
    if error_records:
        logger.warning("Skipping %d invalid chunks", len(error_records))
        write_jsonl(output_dir / "errors.jsonl", error_records)
    if not valid_chunks:
        raise RuntimeError("No valid chunks left to embed")

    actual_device, cuda_available, gpu_name = resolve_device(device)
    logger.info(
        "Resolved device=%s cuda_available=%s gpu_name=%s",
        actual_device,
        cuda_available,
        gpu_name,
    )
    model = load_embedding_model(model_cfg, device=actual_device)

    texts = [ch["embedding_text"] for ch in valid_chunks]
    started_at = _utc_now_iso()
    bs = model.config.batch_size
    n_batches = (len(texts) + bs - 1) // bs
    logger.info(
        "Encoding %d documents in ~%d batches (batch_size=%d) "
        "with model_key=%s model_name=%s",
        len(texts),
        n_batches,
        bs,
        model.config.model_key,
        model.config.model_name,
    )
    sys.stdout.flush()
    t0 = time.perf_counter()
    embeddings = model.encode_documents(
        texts,
        batch_size=bs,
        show_progress_bar=True,
    )
    sys.stdout.flush()
    elapsed = time.perf_counter() - t0
    finished_at = _utc_now_iso()

    if embeddings.ndim != 2 or embeddings.shape[0] != len(valid_chunks):
        raise RuntimeError(
            f"Embedding shape mismatch: got {embeddings.shape}, expected ({len(valid_chunks)}, D)"
        )
    embedding_dim = int(embeddings.shape[1])
    embeddings = embeddings.astype(np.float32, copy=False)

    emb_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.jsonl"
    info_path = output_dir / "run_info.json"

    np.save(emb_path, embeddings)
    write_jsonl(meta_path, valid_chunks)

    run_info = _build_run_info(
        model,
        number_of_chunks=len(valid_chunks),
        embedding_dim=embedding_dim,
        batch_size=model.config.batch_size,
        device=actual_device,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        started_at=started_at,
        finished_at=finished_at,
        total_encoding_time_sec=elapsed,
    )
    write_json(info_path, run_info)

    logger.info(
        "Done. embeddings=%s meta=%s info=%s elapsed=%.2fs (%.4fs/chunk)",
        emb_path,
        meta_path,
        info_path,
        elapsed,
        run_info["avg_encoding_time_per_chunk_sec"],
    )
    return run_info


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build embeddings for a single model into a single output dir."
    )
    parser.add_argument("--model-key", required=True, type=str)
    parser.add_argument(
        "--chunks-path",
        required=True,
        type=str,
        help="Either a JSONL file or a directory containing *.chunks.json (recursive).",
    )
    parser.add_argument("--config", required=True, type=str, help="Path to embedding_models.yaml")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], type=str
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N chunks (debug mode).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing output if compatible (currently best-effort).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Override trust_remote_code from config to True.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Reserved for future parallel data loading; not used currently.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        verbose=args.verbose,
        log_file=out_dir / "build_embeddings.log",
        name="build_embeddings",
    )
    logger.info("Args: %s", vars(args))
    try:
        build_embeddings_for_model(
            model_key=args.model_key,
            chunks_path=args.chunks_path,
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
            resume=args.resume,
            trust_remote_code=True if args.trust_remote_code else None,
        )
    except FileExistsError as e:
        logger.error("%s", e)
        return 2
    except KeyError as e:
        logger.error("Config error: %s", e)
        return 3
    except Exception:
        logger.exception("Failed to build embeddings")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
