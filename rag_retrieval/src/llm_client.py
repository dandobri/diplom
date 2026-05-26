from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .io_utils import read_yaml

logger = logging.getLogger(__name__)

_MOCK_RAG_ANSWER: Dict[str, Any] = {
    "insufficient_information": False,
    "differential_diagnoses": [
        {
            "diagnosis": "Пример диагностической гипотезы (mock)",
            "probability_level": "medium",
            "reasoning": "Mock-объяснение, не использовать клинически.",
            "supporting_findings": ["mock_finding_1"],
            "missing_information": ["анамнез", "лабораторные данные"],
            "citations": [
                {
                    "source_id": "S1",
                    "chunk_id": "MOCK_CHUNK",
                    "document_id": "MOCK_DOC",
                    "section_title": "Mock section",
                    "pages": "1-1",
                }
            ],
        }
    ],
    "recommended_next_steps": [
        {
            "recommendation": "Пример рекомендации (mock).",
            "reasoning": "Mock-обоснование.",
            "citations": [],
        }
    ],
    "red_flags": [],
    "limitations": [
        "MVP-генератор, не для клинического применения.",
        "Это mock-ответ, реальная LLM не вызывалась.",
    ],
    "disclaimer": (
        "Это не окончательный диагноз. Ответ предназначен для предварительной "
        "аналитики и не заменяет консультацию врача. При признаках угрозы жизни — "
        "срочно обратиться за медицинской помощью."
    ),
}


def _mock_response_text(messages: List[Dict[str, Any]]) -> str:
    answer = json.loads(json.dumps(_MOCK_RAG_ANSWER))
    user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    if "[Источник S1]" in user_text:
        chunk_id = _scan_field(user_text, "[Источник S1]", "chunk_id:")
        doc_id = _scan_field(user_text, "[Источник S1]", "document_id:")
        section = _scan_field(user_text, "[Источник S1]", "section_title:")
        pages = _scan_field(user_text, "[Источник S1]", "pages:")
        if answer["differential_diagnoses"] and answer["differential_diagnoses"][0]["citations"]:
            cit = answer["differential_diagnoses"][0]["citations"][0]
            if chunk_id:
                cit["chunk_id"] = chunk_id
            if doc_id:
                cit["document_id"] = doc_id
            if section:
                cit["section_title"] = section
            if pages:
                cit["pages"] = pages
    if "no-RAG" in user_text or "В этом режиме нет внешних источников" in user_text:
        for d in answer["differential_diagnoses"]:
            d["citations"] = []
        for r in answer["recommended_next_steps"]:
            r["citations"] = []
    return json.dumps(answer, ensure_ascii=False)


def _scan_field(text: str, block_marker: str, field_marker: str) -> Optional[str]:
    start = text.find(block_marker)
    if start < 0:
        return None
    fpos = text.find(field_marker, start)
    if fpos < 0:
        return None
    line_end = text.find("\n", fpos)
    if line_end < 0:
        line_end = len(text)
    value = text[fpos + len(field_marker) : line_end].strip()
    return value or None


class LLMClient:
    def __init__(
        self,
        config_path: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        cfg = read_yaml(config_path)
        llm_cfg: Dict[str, Any] = cfg.get("llm", {}) or {}
        providers: Dict[str, Any] = cfg.get("providers", {}) or {}

        prov_key = provider or llm_cfg.get("provider", "openai_compatible")
        prov_cfg: Dict[str, Any] = dict(providers.get(prov_key, {}) or {})

        self.provider_key: str = prov_key
        self.provider_type: str = (prov_cfg.get("type") or prov_key).lower()

        self.model_name: str = (
            model_name
            or llm_cfg.get("model_name")
            or prov_cfg.get("model_name")
            or ""
        )

        base_url_env = llm_cfg.get("base_url_env") or prov_cfg.get("base_url_env")
        base_url_yaml = llm_cfg.get("base_url") or prov_cfg.get("base_url")
        self.base_url: Optional[str] = (
            os.environ.get(base_url_env) if base_url_env else None
        ) or base_url_yaml
        self._base_url_env_name = base_url_env

        self.temperature: float = float(llm_cfg.get("temperature", 0.1))
        self.max_tokens: int = int(llm_cfg.get("max_tokens", 1500))
        self.timeout_sec: float = float(llm_cfg.get("timeout_sec", 120))
        self.request_json_mode: bool = bool(llm_cfg.get("request_json_mode", True))

        api_key_env = llm_cfg.get("api_key_env") or prov_cfg.get("api_key_env")
        self.api_key: Optional[str] = os.environ.get(api_key_env) if api_key_env else None
        self._api_key_env_name = api_key_env

        self._openai_client = None
        logger.info(
            "LLMClient initialised: provider_key=%s type=%s model=%s "
            "base_url=%s (from env=%s) api_key_set=%s json_mode=%s",
            self.provider_key,
            self.provider_type,
            self.model_name,
            self.base_url or "<openai default>",
            base_url_env if (base_url_env and os.environ.get(base_url_env)) else "no",
            bool(self.api_key),
            self.request_json_mode,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        temperature = self.temperature if temperature is None else float(temperature)
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)

        if self.provider_type == "openai":
            text, usage, raw = self._gen_openai(messages, temperature, max_tokens)
        elif self.provider_type == "ollama":
            text, usage, raw = self._gen_ollama(messages, temperature, max_tokens)
        elif self.provider_type == "mock":
            text, usage, raw = self._gen_mock(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type!r}")

        elapsed = time.perf_counter() - t0
        return {
            "text": text,
            "model_name": self.model_name,
            "provider": self.provider_key,
            "provider_type": self.provider_type,
            "usage": usage,
            "raw_response": raw,
            "elapsed_sec": round(elapsed, 4),
        }
    def _ensure_openai(self) -> None:
        if self._openai_client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for openai-compatible provider. "
                "Install: pip install 'openai>=1.30'"
            ) from e
        if not self.api_key and self.provider_type == "openai":
            if self.base_url is None:
                raise RuntimeError(
                    f"OpenAI API key is required when base_url is not set. "
                    f"Set env var {self._api_key_env_name or 'OPENAI_API_KEY'} "
                    f"or задать llm.base_url для self-hosted endpoint."
                )
            else:
                logger.warning(
                    "API key not set (env %s is empty), but base_url=%s is provided. "
                    "Continuing with placeholder key — works only если ваш endpoint не требует auth.",
                    self._api_key_env_name,
                    self.base_url,
                )
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key or "EMPTY",
            "timeout": self.timeout_sec,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._openai_client = OpenAI(**kwargs)

    def _gen_openai(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        self._ensure_openai()
        assert self._openai_client is not None
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.request_json_mode:
            params["response_format"] = {"type": "json_object"}

        try:
            resp = self._openai_client.chat.completions.create(**params)
        except Exception as e:
            if self.request_json_mode and "response_format" in str(e).lower():
                logger.warning("response_format not supported, retrying without it")
                params.pop("response_format", None)
                resp = self._openai_client.chat.completions.create(**params)
            else:
                raise

        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            pass

        usage_obj = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }

        try:
            raw = resp.model_dump()
        except Exception:
            raw = {"id": getattr(resp, "id", None), "model": getattr(resp, "model", None)}
        return text, usage, raw

    def _gen_ollama(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "requests is required for ollama provider. Install: pip install requests"
            ) from e
        url = (self.base_url or "http://localhost:11434").rstrip("/") + "/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.request_json_mode:
            payload["format"] = "json"
        resp = requests.post(url, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        text = ""
        msg = data.get("message") or {}
        if isinstance(msg, dict):
            text = msg.get("content", "") or ""
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "total_tokens": (
                (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
                if data.get("prompt_eval_count") is not None or data.get("eval_count") is not None
                else None
            ),
        }
        return text, usage, data

    def _gen_mock(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        text = _mock_response_text(messages)
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        raw = {"mock": True}
        return text, usage, raw
