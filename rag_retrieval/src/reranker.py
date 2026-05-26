from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io_utils import read_yaml

logger = logging.getLogger(__name__)




@dataclass
class RerankerConfig:
    reranker_key: str
    model_name: str
    backend: str = "sentence_transformers_cross_encoder"
    batch_size: int = 16
    max_length: int = 1024
    trust_remote_code: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RerankerConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(extra=extra, **kwargs)


def load_reranker_config(config_path: str | Path, reranker_key: str) -> Dict[str, Any]:
    cfg = read_yaml(config_path)
    rerankers = cfg.get("rerankers", {})
    if reranker_key not in rerankers:
        available = sorted(rerankers.keys())
        raise KeyError(
            f"Reranker key '{reranker_key}' not found in {config_path}. "
            f"Available: {available}"
        )
    out = dict(rerankers[reranker_key])
    out["reranker_key"] = reranker_key
    return out




def _resolve_device(requested: str) -> Tuple[str, bool, Optional[str]]:
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
            logger.warning("CUDA requested but unavailable — falling back to CPU.")
            return "cpu", cuda_available, gpu_name
        return "cuda", cuda_available, gpu_name
    return ("cuda" if cuda_available else "cpu"), cuda_available, gpu_name




class Reranker:
    def __init__(
        self,
        model_name: str,
        backend: str,
        *,
        reranker_key: str = "",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 1024,
        trust_remote_code: bool = False,
    ) -> None:
        self.reranker_key = reranker_key or model_name
        self.model_name = model_name
        self.backend = backend.lower()
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.trust_remote_code = bool(trust_remote_code)

        self.device, self.cuda_available, self.gpu_name = _resolve_device(device)

        self._model: Any = None  

    

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []

        query = (query or "").strip()
        
        cleaned: List[str] = []
        empty_mask: List[bool] = []
        for p in passages:
            text = (p or "").strip()
            if not text:
                empty_mask.append(True)
                cleaned.append(" ")  
            else:
                empty_mask.append(False)
                cleaned.append(text)

        if self.backend == "sentence_transformers_cross_encoder":
            scores = self._score_st_cross_encoder(query, cleaned)
        elif self.backend == "flag_embedding":
            scores = self._score_flag(query, cleaned)
        else:
            raise ValueError(f"Unknown reranker backend: {self.backend}")

        if len(scores) != len(passages):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for {len(passages)} passages"
            )

        
        out: List[float] = []
        for s, was_empty in zip(scores, empty_mask):
            if was_empty:
                out.append(float("-inf"))
            else:
                out.append(float(s))
        return out

    def score_batched(
        self, query: str, passages: Sequence[str]
    ) -> List[float]:
        
        return self.score(query, passages)

    

    def _ensure_st_cross_encoder(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for cross-encoder backend. "
                "Install: pip install sentence-transformers"
            ) from e
        kwargs: Dict[str, Any] = {
            "device": self.device,
            "max_length": self.max_length,
        }
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        logger.info(
            "Loading CrossEncoder %s on device=%s max_length=%d",
            self.model_name,
            self.device,
            self.max_length,
        )
        self._model = CrossEncoder(self.model_name, **kwargs)

    def _score_st_cross_encoder(self, query: str, passages: Sequence[str]) -> List[float]:
        self._ensure_st_cross_encoder()
        pairs = [(query, p) for p in passages]
        
        scores = self._model.predict(  
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        return scores.tolist()

    def _ensure_flag(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker  
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding is required for flag_embedding backend. "
                "Install: pip install FlagEmbedding"
            ) from e
        use_fp16 = bool(self.cuda_available and self.device == "cuda")
        logger.info(
            "Loading FlagReranker %s (use_fp16=%s, device=%s)",
            self.model_name,
            use_fp16,
            self.device,
        )
        
        try:
            self._model = FlagReranker(
                self.model_name,
                use_fp16=use_fp16,
                devices=[self.device] if self.device != "auto" else None,
                trust_remote_code=self.trust_remote_code,
            )
        except TypeError:
            
            self._model = FlagReranker(self.model_name, use_fp16=use_fp16)

    def _score_flag(self, query: str, passages: Sequence[str]) -> List[float]:
        self._ensure_flag()
        pairs = [[query, p] for p in passages]
        try:
            scores = self._model.compute_score(  
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=False,
            )
        except TypeError:
            
            scores = self._model.compute_score(pairs, normalize=False)  
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        return [float(s) for s in scores]




def build_reranker(cfg_dict: Dict[str, Any], device: str = "auto") -> Reranker:
    cfg = RerankerConfig.from_dict(cfg_dict)
    return Reranker(
        model_name=cfg.model_name,
        backend=cfg.backend,
        reranker_key=cfg.reranker_key,
        device=device,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        trust_remote_code=cfg.trust_remote_code,
    )




def format_passage_for_reranker(meta: Dict[str, Any], max_chars: int = 4000) -> str:
    """Формирует passage в формате, заданном ТЗ.

    "Документ: {document_title}. Раздел: {section_title}. Категория: {label}.
     Страницы: {page_start}-{page_end}. Текст: {text}"

    Если text пустой — берётся embedding_text.
    """
    parts: List[str] = []
    doc_title = meta.get("document_title")
    if doc_title:
        parts.append(f"Документ: {doc_title}")
    section = meta.get("section_title")
    if section:
        parts.append(f"Раздел: {section}")
    label = meta.get("label")
    if label:
        parts.append(f"Категория: {label}")

    ps, pe = meta.get("page_start"), meta.get("page_end")
    if ps is not None and pe is not None:
        parts.append(f"Страницы: {ps}-{pe}")
    elif ps is not None:
        parts.append(f"Страницы: {ps}")

    text = (meta.get("text") or "").strip()
    if not text:
        text = (meta.get("embedding_text") or "").strip()
    parts.append(f"Текст: {text}")

    passage = ". ".join(parts)
    if len(passage) > max_chars:
        passage = passage[:max_chars]
    return passage
