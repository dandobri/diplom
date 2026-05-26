"""Тесты на ``src.pgvector_import``.

Поведение без БД: фикстура ``pg_conn`` делает ``pytest.skip``, если
не установлен ``psycopg`` / ``pgvector`` или Postgres недоступен по
текущим переменным окружения (PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE).

Поднять зависимый Postgres локально::

    docker compose up -d pgvector
    pytest tests/test_pgvector_import.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest


pytest.importorskip("psycopg", reason="psycopg not installed; skipping pgvector tests")
pytest.importorskip("pgvector", reason="pgvector adapter not installed; skipping pgvector tests")

from src.pgvector_import import (  # noqa: E402
    PgvectorConfig,
    connect,
    count_rows,
    create_schema,
    import_embeddings_dir,
    search_nearest,
    upsert_records,
    _row_from_meta,
)


TEST_TABLE = "chunks_test"
EMBED_DIM = 8


def _make_meta(i: int, *, doc: str = "kr1", label: str = "diagnosis") -> Dict[str, Any]:
    return {
        "id": f"chunk-{i:03d}",
        "document_id": doc,
        "chunk_index": i,
        "text": f"Текст чанка {i}.",
        "embedding_text": f"passage: Текст чанка {i}.",
        "document_title": "Тест-документ",
        "section_id": "1",
        "section_title": "Введение",
        "label": label,
        "source": f"docs/{doc}.pdf",
        "page_start": i,
        "page_end": i,
        "specialty": "cardiology",
        "stage": None,
        "term_expansions": {"ИБС": "ишемическая болезнь сердца"},
        "content_hash": f"hash-{i}",
    }


def _unit_vec(seed: int, dim: int = EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) or 1.0
    return v


def _make_record(i: int, *, model: str = "test-model") -> Dict[str, Any]:
    return _row_from_meta(
        _make_meta(i),
        _unit_vec(seed=i),
        model_name=model,
        embedding_dim=EMBED_DIM,
    )


@pytest.fixture(scope="module")
def pg_conn():
    """Подключение к тестовому Postgres. Skip, если БД недоступна."""
    cfg = PgvectorConfig.from_env()
    try:
        conn = connect(cfg)
    except Exception as exc:
        pytest.skip(f"Postgres+pgvector недоступен ({cfg.host}:{cfg.port}): {exc}")
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _clean_table(pg_conn):
    """Перед каждым тестом — чистая таблица фиксированной размерности."""
    with pg_conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{TEST_TABLE}"')
    pg_conn.commit()
    create_schema(pg_conn, embedding_dim=EMBED_DIM, table=TEST_TABLE)
    yield
    with pg_conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{TEST_TABLE}"')
    pg_conn.commit()


class TestSchema:
    def test_vector_extension_installed(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
        assert row is not None, "pgvector extension должно быть установлено connect()"

    def test_table_exists_with_expected_columns(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s ORDER BY column_name",
                (TEST_TABLE,),
            )
            cols = {r[0] for r in cur.fetchall()}
        expected = {
            "id", "document_id", "chunk_index", "text", "embedding_text",
            "document_title", "section_id", "section_title", "label",
            "source", "page_start", "page_end", "specialty", "stage",
            "term_expansions", "content_hash", "embedding_model",
            "embedding_dim", "embedding", "created_at", "updated_at",
        }
        missing = expected - cols
        assert not missing, f"Не хватает колонок: {missing}"

    def test_id_is_primary_key(self, pg_conn):
        records = [_make_record(1), _make_record(1)]
        with pytest.raises(Exception):
            with pg_conn.cursor() as cur:
                cur.execute(
                    f'INSERT INTO "{TEST_TABLE}" '
                    "(id, document_id, text, embedding_text, embedding_model, embedding_dim, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s), (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        records[0]["id"], "d", "t", "et", "m", EMBED_DIM, records[0]["embedding"],
                        records[1]["id"], "d", "t", "et", "m", EMBED_DIM, records[1]["embedding"],
                    ),
                )
        pg_conn.rollback()

    def test_create_schema_is_idempotent(self, pg_conn):
        create_schema(pg_conn, embedding_dim=EMBED_DIM, table=TEST_TABLE)
        create_schema(pg_conn, embedding_dim=EMBED_DIM, table=TEST_TABLE)
        assert count_rows(pg_conn, table=TEST_TABLE) == 0


class TestUpsert:
    def test_insert_count_matches_input(self, pg_conn):
        records = [_make_record(i) for i in range(10)]
        inserted = upsert_records(pg_conn, records, table=TEST_TABLE, batch_size=3)
        assert inserted == 10
        assert count_rows(pg_conn, table=TEST_TABLE) == 10

    def test_upsert_is_idempotent(self, pg_conn):
        records = [_make_record(i) for i in range(5)]
        upsert_records(pg_conn, records, table=TEST_TABLE)
        upsert_records(pg_conn, records, table=TEST_TABLE)
        assert count_rows(pg_conn, table=TEST_TABLE) == 5

    def test_upsert_updates_existing_row(self, pg_conn):
        rec_v1 = _make_record(1, model="model-v1")
        upsert_records(pg_conn, [rec_v1], table=TEST_TABLE)

        rec_v2 = _make_record(1, model="model-v2")
        rec_v2["text"] = "Обновлённый текст"
        upsert_records(pg_conn, [rec_v2], table=TEST_TABLE)

        with pg_conn.cursor() as cur:
            cur.execute(
                f'SELECT text, embedding_model FROM "{TEST_TABLE}" WHERE id = %s',
                (rec_v1["id"],),
            )
            row = cur.fetchone()
        assert row == ("Обновлённый текст", "model-v2")
        assert count_rows(pg_conn, table=TEST_TABLE) == 1

    def test_empty_input_is_noop(self, pg_conn):
        inserted = upsert_records(pg_conn, [], table=TEST_TABLE)
        assert inserted == 0
        assert count_rows(pg_conn, table=TEST_TABLE) == 0

    def test_all_metadata_fields_preserved(self, pg_conn):
        rec = _make_record(7)
        upsert_records(pg_conn, [rec], table=TEST_TABLE)
        with pg_conn.cursor() as cur:
            cur.execute(
                f'SELECT id, document_id, chunk_index, text, embedding_text, '
                f'document_title, section_id, section_title, label, source, '
                f'page_start, page_end, specialty, stage, term_expansions, '
                f'content_hash, embedding_model, embedding_dim '
                f'FROM "{TEST_TABLE}" WHERE id = %s',
                (rec["id"],),
            )
            row = cur.fetchone()
        assert row[0] == rec["id"]
        assert row[1] == rec["document_id"]
        assert row[2] == rec["chunk_index"]
        assert row[3] == rec["text"]
        assert row[4] == rec["embedding_text"]
        assert row[8] == rec["label"]
        assert row[14] == rec["term_expansions"]
        assert row[16] == rec["embedding_model"]
        assert row[17] == EMBED_DIM


class TestVector:
    def test_embedding_dim_matches(self, pg_conn):
        rec = _make_record(1)
        upsert_records(pg_conn, [rec], table=TEST_TABLE)
        with pg_conn.cursor() as cur:
            cur.execute(
                f'SELECT embedding FROM "{TEST_TABLE}" WHERE id = %s',
                (rec["id"],),
            )
            (vec,) = cur.fetchone()
        arr = np.asarray(vec, dtype=np.float32)
        assert arr.shape == (EMBED_DIM,)
        np.testing.assert_allclose(arr, rec["embedding"], atol=1e-5)

    def test_wrong_dim_rejected(self, pg_conn):
        bad_rec = _make_record(1)
        bad_rec["embedding"] = np.zeros(EMBED_DIM + 1, dtype=np.float32)
        with pytest.raises(Exception):
            upsert_records(pg_conn, [bad_rec], table=TEST_TABLE)
        pg_conn.rollback()

    def test_similarity_search_returns_self_first(self, pg_conn):
        records = [_make_record(i) for i in range(5)]
        upsert_records(pg_conn, records, table=TEST_TABLE)

        query = records[2]["embedding"]
        hits = search_nearest(pg_conn, query, table=TEST_TABLE, top_k=3)

        assert len(hits) == 3
        assert hits[0]["id"] == records[2]["id"]
        assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)
        distances = [h["distance"] for h in hits]
        assert distances == sorted(distances)


class TestImportEmbeddingsDir:
    def test_import_from_dir(self, pg_conn, tmp_path: Path):
        n = 6
        emb = np.stack([_unit_vec(seed=i) for i in range(n)]).astype(np.float32)
        np.save(tmp_path / "embeddings.npy", emb)

        with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps(_make_meta(i), ensure_ascii=False) + "\n")

        run_info = {
            "model_key": "tiny_test",
            "model_name": "tiny/test-model",
            "embedding_dim": EMBED_DIM,
            "number_of_chunks": n,
        }
        (tmp_path / "run_info.json").write_text(
            json.dumps(run_info, ensure_ascii=False), encoding="utf-8"
        )

        with pg_conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{TEST_TABLE}"')
        pg_conn.commit()

        stats = import_embeddings_dir(tmp_path, pg_conn, table=TEST_TABLE)
        assert stats["num_records"] == n
        assert stats["embedding_model"] == "tiny/test-model"
        assert stats["embedding_dim"] == EMBED_DIM
        assert count_rows(pg_conn, table=TEST_TABLE) == n

        with pg_conn.cursor() as cur:
            cur.execute(
                f'SELECT DISTINCT embedding_model, embedding_dim FROM "{TEST_TABLE}"'
            )
            rows = cur.fetchall()
        assert rows == [("tiny/test-model", EMBED_DIM)]

    def test_import_mismatched_counts_raises(self, pg_conn, tmp_path: Path):
        emb = np.stack([_unit_vec(seed=i) for i in range(3)]).astype(np.float32)
        np.save(tmp_path / "embeddings.npy", emb)
        with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for i in range(2):
                f.write(json.dumps(_make_meta(i), ensure_ascii=False) + "\n")
        (tmp_path / "run_info.json").write_text(
            json.dumps({"model_name": "x", "embedding_dim": EMBED_DIM}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="!="):
            import_embeddings_dir(tmp_path, pg_conn, table=TEST_TABLE)
