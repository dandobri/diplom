from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_key: str
    model_name: str
    backend: str = "sentence_transformers"
    document_prefix: str = ""
    query_prefix: str = ""
    query_instruction: str = ""
    normalize: bool = True
    batch_size: int = 32
    max_length: int = 512
    trust_remote_code: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(extra=extra, **kwargs)


class EmbeddingModel:
    def __init__(
        self,
        config: ModelConfig,
        device: str = "cpu",
        cuda_available: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        self.cuda_available = cuda_available
        self._st_model = None 
        self._openai_client = None 
        self._embedding_dim: Optional[int] = None

    def encode_documents(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        prefixed = [self._apply_document_prefix(t) for t in texts]
        return self._encode(prefixed, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def encode_queries(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prefixed = [self._apply_query_prefix(t) for t in texts]
        return self._encode(prefixed, batch_size=batch_size, show_progress_bar=show_progress_bar)

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            sample = self._encode(["test"], batch_size=1, show_progress_bar=False)
            self._embedding_dim = int(sample.shape[1])
        return self._embedding_dim

    def _apply_document_prefix(self, text: str) -> str:
        return f"{self.config.document_prefix}{text}" if self.config.document_prefix else text

    def _apply_query_prefix(self, text: str) -> str:
        if self.config.query_instruction:
            return f"{self.config.query_instruction}{text}"
        if self.config.query_prefix:
            return f"{self.config.query_prefix}{text}"
        return text

    def _encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        bs = batch_size or self.config.batch_size
        backend = self.config.backend.lower()
        if backend == "sentence_transformers":
            return self._encode_st(texts, batch_size=bs, show_progress_bar=show_progress_bar)
        if backend == "openai":
            return self._encode_openai(texts, batch_size=bs, show_progress_bar=show_progress_bar)
        raise ValueError(f"Unknown backend: {self.config.backend}")

    def _ensure_st(self) -> None:
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Install it: pip install sentence-transformers"
            ) from e

        kwargs: Dict[str, Any] = {"device": self.device}
        if self.config.trust_remote_code:
            kwargs["trust_remote_code"] = True
        logger.info(
            "Loading sentence-transformers model %s on device %s",
            self.config.model_name,
            self.device,
        )
        self._st_model = SentenceTransformer(self.config.model_name, **kwargs)
        try:
            self._st_model.max_seq_length = self.config.max_length
        except Exception:
            logger.debug("Could not set max_seq_length on model")

    def _encode_st(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        self._ensure_st()
        assert self._st_model is not None
        embeddings = self._st_model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _ensure_openai(self) -> None:
        if self._openai_client is not None:
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. OpenAI backend is optional and only "
                "available when the env var is configured."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for openai backend. Install with: pip install openai"
            ) from e
        self._openai_client = OpenAI(api_key=api_key)

    def _encode_openai(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        self._ensure_openai()
        assert self._openai_client is not None

        from tqdm.auto import tqdm

        all_vecs: List[List[float]] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc=f"openai:{self.config.model_name}")
        for start in iterator:
            batch = list(texts[start : start + batch_size])
            resp = self._openai_client.embeddings.create(
                model=self.config.model_name,
                input=batch,
            )
            for item in resp.data:
                all_vecs.append(item.embedding)
        arr = np.asarray(all_vecs, dtype=np.float32)
        if self.config.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


def resolve_device(requested: str = "auto") -> tuple[str, bool, Optional[str]]:
    cuda_available = False
    gpu_name: Optional[str] = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
    except ImportError:
        cuda_available = False

    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu", cuda_available, gpu_name
    if requested == "cuda":
        if not cuda_available:
            logger.warning(
                "CUDA was requested but is not available. Falling back to CPU."
            )
            return "cpu", cuda_available, gpu_name
        return "cuda", cuda_available, gpu_name
    return ("cuda" if cuda_available else "cpu"), cuda_available, gpu_name


def load_embedding_model(
    model_cfg_dict: Dict[str, Any],
    device: str = "auto",
) -> EmbeddingModel:
    cfg = ModelConfig.from_dict(model_cfg_dict)
    actual_device, cuda_available, _gpu = resolve_device(device)
    return EmbeddingModel(config=cfg, device=actual_device, cuda_available=cuda_available)
