"""Импорт эмбеддингов и метаданных чанков в Postgres + pgvector.

Дополняет ``pgvector_export.py`` (тот пишет JSONL+SQL на диск).
Здесь — реальное подключение к Postgres, создание схемы и UPSERT
по полю ``id``.

CLI::

    python -m src.pgvector_import \\
        --embeddings-dir outputs/embeddings/e5_large \\
        --table chunks

Параметры подключения читаются из окружения (PGHOST/PGPORT/PGUSER/
PGPASSWORD/PGDATABASE) с дефолтами под локальный docker-compose.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .io_utils import read_json, read_jsonl_list

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


DEFAULT_TABLE = "chunks"


@dataclass(frozen=True)
class PgvectorConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "rag"
    password: str = "rag"
    dbname: str = "rag"

    @classmethod
    def from_env(cls) -> "PgvectorConfig":
        return cls(
            host=os.environ.get("PGHOST", "localhost"),
            port=int(os.environ.get("PGPORT", "5432")),
            user=os.environ.get("PGUSER", "rag"),
            password=os.environ.get("PGPASSWORD", "rag"),
            dbname=os.environ.get("PGDATABASE", "rag"),
        )

    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={self.dbname}"
        )


def connect(config: Optional[PgvectorConfig] = None):
    """Открыть соединение и зарегистрировать тип ``vector``."""
    import psycopg
    from pgvector.psycopg import register_vector

    cfg = config or PgvectorConfig.from_env()
    conn = psycopg.connect(cfg.conninfo())
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def _quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def create_schema(
    conn,
    *,
    embedding_dim: int,
    table: str = DEFAULT_TABLE,
) -> None:
    """Создать таблицу chunks и метаданные-индексы (идемпотентно)."""
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
    tbl = _quote_ident(table)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {tbl} (
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
        embedding       vector({embedding_dim}) NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
    indexes = [
        f"CREATE INDEX IF NOT EXISTS {table}_document_id_idx ON {tbl} (document_id)",
        f"CREATE INDEX IF NOT EXISTS {table}_label_idx        ON {tbl} (label)",
        f"CREATE INDEX IF NOT EXISTS {table}_specialty_idx    ON {tbl} (specialty)",
        f"CREATE INDEX IF NOT EXISTS {table}_content_hash_idx ON {tbl} (content_hash)",
    ]
    with conn.cursor() as cur:
        cur.execute(ddl)
        for stmt in indexes:
            cur.execute(stmt)
    conn.commit()
    logger.info("Schema ready: table=%s, embedding_dim=%d", table, embedding_dim)


def _row_from_meta(
    meta: Dict[str, Any],
    embedding: np.ndarray,
    *,
    model_name: str,
    embedding_dim: int,
) -> Dict[str, Any]:
    row = {field: meta.get(field) for field in METADATA_FIELDS_ORDER}
    if not row.get("id"):
        raise ValueError("metadata row missing required field 'id'")
    row["embedding_model"] = model_name
    row["embedding_dim"] = int(embedding_dim)
    row["embedding"] = np.asarray(embedding, dtype=np.float32)
    return row


def _upsert_sql(table: str) -> str:
    tbl = _quote_ident(table)
    cols = list(METADATA_FIELDS_ORDER) + ["embedding_model", "embedding_dim", "embedding"]
    col_list = ", ".join(_quote_ident(c) for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))
    update_cols = [c for c in cols if c != "id"]
    update_set = ", ".join(
        f"{_quote_ident(c)} = EXCLUDED.{_quote_ident(c)}" for c in update_cols
    )
    return (
        f"INSERT INTO {tbl} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT (id) DO UPDATE SET {update_set}, updated_at = NOW()"
    )


def _row_to_params(row: Dict[str, Any]) -> tuple:
    from psycopg.types.json import Jsonb

    out: List[Any] = []
    for col in list(METADATA_FIELDS_ORDER) + ["embedding_model", "embedding_dim", "embedding"]:
        val = row.get(col)
        if col == "term_expansions" and val is not None:
            val = Jsonb(val)
        out.append(val)
    return tuple(out)


def upsert_records(
    conn,
    records: Sequence[Dict[str, Any]],
    *,
    table: str = DEFAULT_TABLE,
    batch_size: int = 500,
) -> int:
    """UPSERT-ит готовые записи (dict со всеми колонками + ``embedding`` как np.ndarray)."""
    if not records:
        return 0
    sql = _upsert_sql(table)
    total = 0
    with conn.cursor() as cur:
        batch: List[tuple] = []
        for rec in records:
            batch.append(_row_to_params(rec))
            if len(batch) >= batch_size:
                cur.executemany(sql, batch)
                total += len(batch)
                batch = []
        if batch:
            cur.executemany(sql, batch)
            total += len(batch)
    conn.commit()
    logger.info("Upserted %d records into %s", total, table)
    return total


def import_embeddings_dir(
    embeddings_dir: str | Path,
    conn,
    *,
    table: str = DEFAULT_TABLE,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Прочитать `embeddings.npy` + `metadata.jsonl` + `run_info.json` и залить в БД."""
    embeddings_dir = Path(embeddings_dir)
    emb_path = embeddings_dir / "embeddings.npy"
    meta_path = embeddings_dir / "metadata.jsonl"
    info_path = embeddings_dir / "run_info.json"
    for p in (emb_path, meta_path, info_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    embeddings = np.load(emb_path)
    metadata = read_jsonl_list(meta_path)
    run_info = read_json(info_path)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != metadata rows ({len(metadata)})"
        )

    model_name = run_info.get("model_name") or run_info.get("model_key") or "unknown"
    embedding_dim = int(run_info.get("embedding_dim", embeddings.shape[1]))
    if embedding_dim != embeddings.shape[1]:
        raise ValueError(
            f"run_info.embedding_dim={embedding_dim} != embeddings.shape[1]={embeddings.shape[1]}"
        )

    create_schema(conn, embedding_dim=embedding_dim, table=table)

    records = [
        _row_from_meta(meta, vec, model_name=model_name, embedding_dim=embedding_dim)
        for meta, vec in zip(metadata, embeddings)
    ]
    inserted = upsert_records(conn, records, table=table, batch_size=batch_size)

    return {
        "table": table,
        "embedding_model": model_name,
        "embedding_dim": embedding_dim,
        "num_records": inserted,
        "embeddings_dir": str(embeddings_dir),
    }


def count_rows(conn, *, table: str = DEFAULT_TABLE) -> int:
    tbl = _quote_ident(table)
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        row = cur.fetchone()
    return int(row[0]) if row else 0


def search_nearest(
    conn,
    query_vector: np.ndarray,
    *,
    table: str = DEFAULT_TABLE,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Cosine-similarity поиск через pgvector (`<=>`).

    Возвращает список `{id, document_id, label, distance}` отсортированный
    по возрастанию расстояния (т.е. сначала самые похожие).
    """
    tbl = _quote_ident(table)
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    sql = (
        f"SELECT id, document_id, label, embedding <=> %s AS distance "
        f"FROM {tbl} ORDER BY embedding <=> %s LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (q, q, int(top_k)))
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "document_id": r[1],
            "label": r[2],
            "distance": float(r[3]),
        }
        for r in rows
    ]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import embeddings + metadata into Postgres/pgvector"
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        type=str,
        help="Path to outputs/embeddings/<model_key>",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE, type=str)
    parser.add_argument("--batch-size", default=500, type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        conn = connect()
    except Exception:
        logger.exception("Cannot connect to Postgres")
        return 2
    try:
        stats = import_embeddings_dir(
            args.embeddings_dir, conn, table=args.table, batch_size=args.batch_size
        )
        logger.info("Done: %s", stats)
    except Exception:
        logger.exception("Import failed")
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
