from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .io_utils import ensure_dir, read_jsonl_list, write_json, safe_get

logger = logging.getLogger(__name__)

CHUNK_FILE_PATTERN = "*.chunks.json"

REQUIRED_FIELDS: Tuple[str, ...] = (
    "id",
    "document_id",
    "chunk_index",
    "text",
    "embedding_text",
    "document_title",
    "section_title",
    "label",
    "source",
    "page_start",
    "page_end",
    "content_hash",
)


def discover_chunk_files(chunks_dir: Path) -> List[Path]:
    
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory does not exist: {chunks_dir}")

    if not chunks_dir.is_dir():
        raise NotADirectoryError(f"Chunks path is not a directory: {chunks_dir}")

    return sorted(
        path
        for path in chunks_dir.rglob(CHUNK_FILE_PATTERN)
        if path.is_file()
    )


def _read_json_chunks(path: Path) -> List[Dict[str, Any]]:
    """Читает .chunks.json файл.

    Поддерживаемые форматы:
        1. JSON-массив:
            [{...}, {...}]

        2. JSON-объект с ключом chunks:
            {"chunks": [{...}, {...}]}

        3. JSONL fallback:
            {...}
            {...}
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            chunks = data
        elif isinstance(data, dict) and isinstance(data.get("chunks"), list):
            chunks = data["chunks"]
        else:
            raise ValueError(
                "expected JSON array or JSON object with key 'chunks'"
            )

    except json.JSONDecodeError:
        
        
        chunks = read_jsonl_list(str(path))

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"chunk #{idx} in {path} is not a JSON object"
            )

    return chunks


def load_chunks_from_dir(
    chunks_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, str]]]:
    """Загружает чанки из всех *.chunks.json файлов в директории.

    Returns:
        Кортеж:
            - список чанков
            - количество чанков по файлам
            - ошибки чтения файлов
    """
    chunk_files = discover_chunk_files(chunks_dir)

    chunks: List[Dict[str, Any]] = []
    chunks_by_file: Dict[str, int] = {}
    file_errors: List[Dict[str, str]] = []

    if not chunk_files:
        file_errors.append(
            {
                "file": str(chunks_dir),
                "error": f"no files matching '**/{CHUNK_FILE_PATTERN}'",
            }
        )
        return chunks, chunks_by_file, file_errors

    for path in chunk_files:
        relative_path = str(path.relative_to(chunks_dir))

        try:
            file_chunks = _read_json_chunks(path)
        except Exception as exc:
            file_errors.append(
                {
                    "file": relative_path,
                    "error": str(exc),
                }
            )
            continue

        chunks_by_file[relative_path] = len(file_chunks)

        for chunk in file_chunks:
            
            chunk_with_meta = dict(chunk)
            chunk_with_meta["_source_file"] = relative_path
            chunks.append(chunk_with_meta)

    return chunks, chunks_by_file, file_errors


def _validate_chunk(chunk: Dict[str, Any]) -> List[str]:
    """Возвращает список ошибок для одного чанка.

    Пустой список означает, что чанк валиден.
    """
    errors: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in chunk:
            errors.append(f"missing field '{field}'")
            continue

        value = chunk[field]

        if value is None:
            errors.append(f"field '{field}' is null")
            continue

        if field in ("text", "embedding_text"):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"field '{field}' is empty")

    return errors


def validate_chunks(
    chunks: List[Dict[str, Any]],
    *,
    chunks_by_file: Dict[str, int] | None = None,
    file_errors: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Запускает все проверки и возвращает агрегированный отчет.

    Args:
        chunks: список словарей-чанков.
        chunks_by_file: количество чанков по каждому файлу.
        file_errors: ошибки чтения отдельных файлов.

    Returns:
        Словарь с ключами:
            - is_valid
            - total_chunks
            - errors
            - file_errors
            - duplicate_ids
            - duplicate_content_hashes
            - statistics
    """
    chunks_by_file = chunks_by_file or {}
    file_errors = file_errors or []

    errors_per_chunk: List[Dict[str, Any]] = []

    id_counter: Counter = Counter()
    hash_counter: Counter = Counter()

    text_lengths: List[int] = []
    embedding_text_lengths: List[int] = []

    by_document: Counter = Counter()
    by_label: Counter = Counter()
    by_specialty: Counter = Counter()

    for idx, chunk in enumerate(chunks):
        errs = _validate_chunk(chunk)

        if errs:
            errors_per_chunk.append(
                {
                    "index": idx,
                    "file": safe_get(chunk, "_source_file"),
                    "id": safe_get(chunk, "id"),
                    "errors": errs,
                }
            )

        chunk_id = safe_get(chunk, "id")
        if chunk_id is not None:
            id_counter[chunk_id] += 1

        content_hash = safe_get(chunk, "content_hash")
        if content_hash is not None:
            hash_counter[content_hash] += 1

        text = safe_get(chunk, "text") or ""
        embedding_text = safe_get(chunk, "embedding_text") or ""

        text_lengths.append(len(text))
        embedding_text_lengths.append(len(embedding_text))

        doc_id = safe_get(chunk, "document_id")
        if doc_id is not None:
            by_document[doc_id] += 1

        label = safe_get(chunk, "label")
        if label is not None:
            by_label[label] += 1

        specialty = safe_get(chunk, "specialty")
        if specialty is not None:
            by_specialty[specialty] += 1

    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}
    duplicate_hashes = {k: v for k, v in hash_counter.items() if v > 1}

    def _avg(values: List[int]) -> float:
        return float(statistics.mean(values)) if values else 0.0

    stats = {
        "total_files": len(chunks_by_file),
        "total_chunks": len(chunks),
        "unique_documents": len(by_document),
        "chunks_by_file": chunks_by_file,
        "chunks_by_document": dict(by_document.most_common()),
        "chunks_by_label": dict(by_label.most_common()),
        "chunks_by_specialty": dict(by_specialty.most_common()),
        "avg_text_length": _avg(text_lengths),
        "avg_embedding_text_length": _avg(embedding_text_lengths),
        "min_text_length": min(text_lengths) if text_lengths else 0,
        "max_text_length": max(text_lengths) if text_lengths else 0,
        "min_embedding_text_length": (
            min(embedding_text_lengths) if embedding_text_lengths else 0
        ),
        "max_embedding_text_length": (
            max(embedding_text_lengths) if embedding_text_lengths else 0
        ),
    }

    is_valid = (
        not file_errors
        and not errors_per_chunk
        and not duplicate_ids
        and not duplicate_hashes
    )

    return {
        "is_valid": is_valid,
        "total_chunks": len(chunks),
        "errors": errors_per_chunk,
        "file_errors": file_errors,
        "duplicate_ids": duplicate_ids,
        "duplicate_content_hashes": duplicate_hashes,
        "statistics": stats,
    }


def save_report_csv(report: Dict[str, Any], path: Path) -> None:
    """Сохраняет краткую таблицу со статистиками в CSV.

    Используем ручную запись, чтобы не зависеть от pandas в этом модуле.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    stats = report["statistics"]

    rows = [
        ("metric", "value"),
        ("is_valid", report["is_valid"]),
        ("total_files", stats["total_files"]),
        ("total_chunks", stats["total_chunks"]),
        ("unique_documents", stats["unique_documents"]),
        ("avg_text_length", round(stats["avg_text_length"], 2)),
        ("avg_embedding_text_length", round(stats["avg_embedding_text_length"], 2)),
        ("min_text_length", stats["min_text_length"]),
        ("max_text_length", stats["max_text_length"]),
        ("min_embedding_text_length", stats["min_embedding_text_length"]),
        ("max_embedding_text_length", stats["max_embedding_text_length"]),
        ("num_file_errors", len(report["file_errors"])),
        ("num_chunk_errors", len(report["errors"])),
        ("num_duplicate_ids", len(report["duplicate_ids"])),
        ("num_duplicate_content_hashes", len(report["duplicate_content_hashes"])),
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerows(rows)

        writer.writerow([])
        writer.writerow(["chunks_by_file", ""])
        for file_path, count in stats["chunks_by_file"].items():
            writer.writerow([file_path, count])

        writer.writerow([])
        writer.writerow(["chunks_by_label", ""])
        for label, count in stats["chunks_by_label"].items():
            writer.writerow([label, count])

        writer.writerow([])
        writer.writerow(["chunks_by_specialty", ""])
        for specialty, count in stats["chunks_by_specialty"].items():
            writer.writerow([specialty, count])

        writer.writerow([])
        writer.writerow(["file_errors", ""])
        for error in report["file_errors"]:
            writer.writerow([error["file"], error["error"]])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate all *.chunks.json files recursively"
    )
    parser.add_argument("--chunks-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    chunks_dir = Path(args.chunks_dir)

    chunks, chunks_by_file, file_errors = load_chunks_from_dir(chunks_dir)

    logger.info(
        "Loaded %d chunks from %d files under %s",
        len(chunks),
        len(chunks_by_file),
        chunks_dir,
    )

    if file_errors:
        logger.warning("Found %d file read errors", len(file_errors))

    report = validate_chunks(
        chunks,
        chunks_by_file=chunks_by_file,
        file_errors=file_errors,
    )

    out_dir = ensure_dir(args.output_dir)

    json_path = out_dir / "chunk_validation_report.json"
    csv_path = out_dir / "chunk_validation_report.csv"

    write_json(json_path, report)
    save_report_csv(report, csv_path)

    logger.info("Validation: is_valid=%s", report["is_valid"])
    logger.info("Total files: %d", report["statistics"]["total_files"])
    logger.info("Total chunks: %d", report["total_chunks"])
    logger.info("Unique documents: %d", report["statistics"]["unique_documents"])
    logger.info(
        "Avg text length: %.1f, avg embedding_text length: %.1f",
        report["statistics"]["avg_text_length"],
        report["statistics"]["avg_embedding_text_length"],
    )

    if report["file_errors"]:
        logger.warning("Found %d file errors", len(report["file_errors"]))

    if report["errors"]:
        logger.warning("Found %d invalid chunks", len(report["errors"]))

    if report["duplicate_ids"]:
        logger.warning("Found %d duplicate ids", len(report["duplicate_ids"]))

    if report["duplicate_content_hashes"]:
        logger.warning(
            "Found %d duplicate content_hashes",
            len(report["duplicate_content_hashes"]),
        )

    logger.info("Report saved to %s and %s", json_path, csv_path)

    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())