from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import read_yaml




SYSTEM_PROMPT_MEDICAL_RAG = """\
Ты — медицинский AI-ассистент для предварительной аналитики, работающий поверх \
выдержек из российских клинических рекомендаций. Ты не ставишь окончательный \
диагноз — только формируешь дифференциальные диагностические гипотезы и \
рекомендации по дальнейшим шагам.

Жёсткие правила:
1. В RAG-режиме используй ТОЛЬКО переданные клинические фрагменты ([Источник S1], \
[Источник S2]…). Не выдумывай факты вне источников.
2. Каждое существенное медицинское утверждение в differential_diagnoses и \
recommended_next_steps должно иметь хотя бы одну citation на конкретный source_id \
из переданного контекста.
3. Не ссылайся на document_id / chunk_id / страницы, которых нет в контексте.
4. Не ставь окончательный диагноз. Используй вероятностные градации, но не точные \
проценты.
5. probability_level — только одно из: "high", "medium", "low", "unknown".
6. Если переданных данных недостаточно для какого-либо вывода — выставь \
insufficient_information=true и опиши, чего не хватает, в limitations / \
missing_information.
7. Если источник прямо противоречит гипотезе — не включай эту гипотезу в \
differential_diagnoses.

Безопасность (обязательно отражать в disclaimer):
- Это не окончательный диагноз.
- Ответ предназначен для предварительной аналитики и не заменяет консультацию врача.
- При признаках угрозы жизни — срочно обратиться за медицинской помощью.

Формат ответа: СТРОГО валидный JSON по заданной схеме. Никакого текста до или после \
JSON, никаких markdown-блоков. Если ты не уверен — сначала загерируй полный JSON, \
проверь его и только потом верни.
"""


JSON_SCHEMA_TEXT = """\
{
  "insufficient_information": false,
  "differential_diagnoses": [
    {
      "diagnosis": "string",
      "probability_level": "high|medium|low|unknown",
      "reasoning": "string",
      "supporting_findings": ["string"],
      "missing_information": ["string"],
      "citations": [
        {
          "source_id": "S1",
          "chunk_id": "string",
          "document_id": "string",
          "section_title": "string",
          "pages": "12-13"
        }
      ]
    }
  ],
  "recommended_next_steps": [
    {
      "recommendation": "string",
      "reasoning": "string",
      "citations": [
        {
          "source_id": "S2",
          "chunk_id": "string",
          "document_id": "string",
          "section_title": "string",
          "pages": "15"
        }
      ]
    }
  ],
  "red_flags": [
    {
      "red_flag": "string",
      "action": "string",
      "citations": []
    }
  ],
  "limitations": ["string"],
  "disclaimer": "string"
}
"""


RAG_DIAGNOSTIC_PROMPT = """\
Клинический случай (описание пациента):
\"\"\"
{patient_case}
\"\"\"

Найденные фрагменты клинических рекомендаций (используй ТОЛЬКО их):
{retrieved_context}

Сформируй ответ строго в JSON по схеме:
{json_schema}

Напоминание:
- Не используй citations, которых нет среди источников выше.
- source_id должен быть одним из S1..S{num_sources}.
- chunk_id и document_id в citation должны точно совпадать с теми, что в источнике.
- Не ставь окончательный диагноз.
- При недостатке данных — insufficient_information=true.
"""


NO_RAG_DIAGNOSTIC_PROMPT = """\
Клинический случай (описание пациента):
\"\"\"
{patient_case}
\"\"\"

ВНИМАНИЕ: это baseline без RAG-источников.
В этом режиме нет внешних источников. Поле "citations" должно быть пустым массивом \
для всех элементов. Не выдумывай источники, document_id, chunk_id, страницы.

Сформируй ответ строго в JSON по схеме:
{json_schema}

Напоминание:
- Не выдумывай источники.
- Если без источников нельзя сформулировать гипотезу — insufficient_information=true.
- Не ставь окончательный диагноз.
"""




REQUIRED_PROMPT_FIELDS = ("system_prompt", "user_prompt_template")


def load_prompt_config(path: str | Path) -> Dict[str, Any]:
    """Загружает prompt-config из YAML.

    Обязательные поля: ``system_prompt``, ``user_prompt_template``.
    Опциональные: ``name``, ``version``, ``language``, ``json_schema``,
    ``rules``.
    """
    cfg = read_yaml(path)
    missing = [k for k in REQUIRED_PROMPT_FIELDS if not cfg.get(k)]
    if missing:
        raise ValueError(
            f"Prompt config {path} is missing required fields: {missing}"
        )
    cfg.setdefault("json_schema", JSON_SCHEMA_TEXT)
    return cfg


class _SafeFormatDict(dict):
    """Возвращает плейсхолдер вместо KeyError, чтобы шаблон не падал на
    неизвестных ``{...}`` в чужом prompt-config."""

    def __missing__(self, key: str) -> str:  
        return "{" + key + "}"


def build_messages_from_prompt_config(
    prompt_config: Dict[str, Any],
    patient_case: str,
    retrieved_context: Optional[str] = None,
    num_sources: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Собирает messages для chat.completions.create() из YAML-prompt'а.

    Подставляются плейсхолдеры:
        {patient_case}
        {retrieved_context}     — пустая строка для no-RAG
        {json_schema}
        {num_sources}           — 0 для no-RAG
    Лишние плейсхолдеры в шаблоне не ломают рендер.
    """
    system_prompt = (prompt_config.get("system_prompt") or "").strip()
    template = prompt_config.get("user_prompt_template") or ""
    schema = prompt_config.get("json_schema") or JSON_SCHEMA_TEXT

    fmt = _SafeFormatDict(
        patient_case=(patient_case or "").strip(),
        retrieved_context=(retrieved_context or "").strip(),
        json_schema=schema,
        num_sources=int(num_sources) if num_sources is not None else 0,
    )
    user = template.format_map(fmt)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def build_messages_from_judge_prompt_config(
    prompt_config: Dict[str, Any],
    **placeholders: Any,
) -> List[Dict[str, str]]:
    """Универсальный builder для judge-prompts.

    В отличие от ``build_messages_from_prompt_config`` не предполагает наличие
    обязательных полей ``patient_case`` / ``retrieved_context`` — все
    плейсхолдеры произвольные. ``json_schema`` всегда подставляется из
    prompt_config (или из default ``JSON_SCHEMA_TEXT``), если шаблон его
    использует.

    Пример::

        msgs = build_messages_from_judge_prompt_config(
            prompt_cfg,
            case_id="case001", mode="rag",
            patient_case="...",
            reference_context="[Источник S1]\\n...",
            claims_json="[ ... ]",
        )
    """
    system_prompt = (prompt_config.get("system_prompt") or "").strip()
    template = prompt_config.get("user_prompt_template") or ""
    schema = prompt_config.get("json_schema") or JSON_SCHEMA_TEXT

    
    fmt_kwargs: Dict[str, Any] = {
        "json_schema": schema,
    }
    for k, v in placeholders.items():
        if v is None:
            fmt_kwargs[k] = ""
        elif isinstance(v, str):
            fmt_kwargs[k] = v
        else:
            fmt_kwargs[k] = str(v)

    user = template.format_map(_SafeFormatDict(fmt_kwargs))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]




def build_rag_messages(
    patient_case: str,
    retrieved_context: str,
    num_sources: int,
    *,
    prompt_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    
    if prompt_config is not None:
        return build_messages_from_prompt_config(
            prompt_config,
            patient_case=patient_case,
            retrieved_context=retrieved_context,
            num_sources=num_sources,
        )
    user = RAG_DIAGNOSTIC_PROMPT.format(
        patient_case=(patient_case or "").strip(),
        retrieved_context=(retrieved_context or "").strip(),
        json_schema=JSON_SCHEMA_TEXT,
        num_sources=max(1, num_sources),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_MEDICAL_RAG},
        {"role": "user", "content": user},
    ]


def build_no_rag_messages(
    patient_case: str,
    *,
    prompt_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    
    if prompt_config is not None:
        return build_messages_from_prompt_config(
            prompt_config,
            patient_case=patient_case,
            retrieved_context=None,
            num_sources=0,
        )
    user = NO_RAG_DIAGNOSTIC_PROMPT.format(
        patient_case=(patient_case or "").strip(),
        json_schema=JSON_SCHEMA_TEXT,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_MEDICAL_RAG},
        {"role": "user", "content": user},
    ]
