from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .io_utils import ensure_dir, read_json, read_jsonl_list, write_jsonl

logger = logging.getLogger(__name__)


METADATA_FIELDS_ORDER = (
    "id",
    "document_id",
    "chunk_index",
    "text",
    "embedding_text",
    "document_title",
    "section_id",
    "section_title",
    "label",
    "source",
    "page_start",
    "page_end",
    "specialty",
    "stage",
    "term_expansions",
    "content_hash",
)


def build_export_records(
    metadata: List[Dict[str, Any]],
    embeddings: np.ndarray,
    *,
    model_name: str,
    embedding_dim: int,
) -> List[Dict[str, Any]]:
    
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != metadata rows ({len(metadata)})"
        )
    out: List[Dict[str, Any]] = []
    for row, vec in zip(metadata, embeddings):
        rec: Dict[str, Any] = {}
        for field in METADATA_FIELDS_ORDER:
            rec[field] = row.get(field)
        rec["embedding"] = vec.astype(float).tolist()
        rec["embedding_model"] = model_name
        rec["embedding_dim"] = int(embedding_dim)
        out.append(rec)
    return out


SQL_TEMPLATE = """\
-- pgvector schema for model: {model_key} ({model_name})
-- Embedding dimension: {dim}
--
-- Run once per database:
--   CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    chunk_index     INTEGER,
    text            TEXT NOT NULL,
    embedding_text  TEXT NOT NULL,
    document_title  TEXT,
    section_id      TEXT,
    section_title   TEXT,
    label           TEXT,
    source          TEXT,
    page_start      INTEGER,
    page_end        INTEGER,
    specialty       TEXT,
    stage           TEXT,
    term_expansions JSONB,
    content_hash    TEXT,
    embedding_model TEXT NOT NULL,
    embedding_dim   INTEGER NOT NULL,
    embedding       vector({dim}) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Метаданные для фильтрации
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_label_idx        ON chunks (label);
CREATE INDEX IF NOT EXISTS chunks_specialty_idx    ON chunks (specialty);
CREATE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks (content_hash);

-- Векторный индекс. Используем cosine, т.к. embeddings нормализованы.
-- IVFFLAT — для корпусов до ~1M; для больших можно заменить на HNSW.
-- ВАЖНО: создавать индекс после загрузки данных, чтобы lists подобрался корректно.
-- CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat_idx
--   ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--
-- Альтернатива (HNSW, требует pgvector 0.5+):
-- CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
--   ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def write_sql_schema(path: Path, *, model_key: str, model_name: str, dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sql = SQL_TEMPLATE.format(model_key=model_key, model_name=model_name, dim=int(dim))
    path.write_text(sql, encoding="utf-8")


def export_model_to_pgvector(
    *,
    model_key: str,
    embeddings_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    embeddings_dir = Path(embeddings_dir)
    output_dir = ensure_dir(output_dir)

    emb_path = embeddings_dir / "embeddings.npy"
    meta_path = embeddings_dir / "metadata.jsonl"
    info_path = embeddings_dir / "run_info.json"
    for p in (emb_path, meta_path, info_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    embeddings = np.load(emb_path)
    metadata = read_jsonl_list(meta_path)
    run_info = read_json(info_path)

    model_name = run_info.get("model_name", model_key)
    embedding_dim = int(run_info.get("embedding_dim", embeddings.shape[1]))

    records = build_export_records(
        metadata,
        embeddings,
        model_name=model_name,
        embedding_dim=embedding_dim,
    )

    jsonl_path = output_dir / f"chunks_for_pgvector_{model_key}.jsonl"
    sql_path = output_dir / f"schema_{model_key}.sql"
    write_jsonl(jsonl_path, records)
    write_sql_schema(sql_path, model_key=model_key, model_name=model_name, dim=embedding_dim)

    logger.info("Wrote %d records to %s", len(records), jsonl_path)
    logger.info("Wrote SQL schema to %s", sql_path)

    return {
        "model_key": model_key,
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_records": len(records),
        "jsonl_path": str(jsonl_path),
        "sql_path": str(sql_path),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pgvector export for the chosen model")
    parser.add_argument("--model-key", required=True, type=str)
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        type=str,
        help="Path to outputs/embeddings/<model_key>",
    )
    parser.add_argument("--output-dir", default="outputs/pgvector", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        export_model_to_pgvector(
            model_key=args.model_key,
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
        )
    except Exception: 
        logger.exception("Export failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
