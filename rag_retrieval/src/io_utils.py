from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    *,
    verbose: bool = False,
    log_file: Optional[str | Path] = None,
    name: str = "build",
) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True) 
    except Exception:
        pass
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        logging.getLogger(__name__).info("Logging to file: %s", log_path)


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Failed to parse JSONL at {p}:{line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e


def read_jsonl_list(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def write_jsonl(path: str | Path, items: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
    return data


def load_model_config(config_path: str | Path, model_key: str) -> Dict[str, Any]:
    cfg = read_yaml(config_path)
    models = cfg.get("models", {})
    if model_key not in models:
        available = sorted(models.keys())
        raise KeyError(
            f"Model key '{model_key}' not found in {config_path}. "
            f"Available: {available}"
        )
    model_cfg = dict(models[model_key])
    model_cfg["model_key"] = model_key
    return model_cfg


def list_model_keys(config_path: str | Path) -> List[str]:
    cfg = read_yaml(config_path)
    return list(cfg.get("models", {}).keys())


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_get(d: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if d is None:
        return default
    return d.get(key, default)


def load_chunks_any(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks path not found: {p}")

    if p.is_dir():
        from .validate_chunks import load_chunks_from_dir

        chunks, _by_file, file_errors = load_chunks_from_dir(p)
        if file_errors:
            for fe in file_errors:
                logger.warning("Chunk file error: %s — %s", fe.get("file"), fe.get("error"))
        return chunks

    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl_list(p)

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        return list(data["chunks"])
    raise ValueError(
        f"Unsupported JSON layout in {p}: expected list or object with 'chunks' key"
    )
