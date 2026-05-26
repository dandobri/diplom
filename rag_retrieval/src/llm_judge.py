from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from .llm_client import LLMClient
from .prompt_templates import build_messages_from_judge_prompt_config
from .rag_generation import parse_llm_json

logger = logging.getLogger(__name__)






CLAIM_TYPES = (
    "diagnosis",
    "reasoning",
    "finding",
    "missing_information",
    "recommendation",
    "rec_reasoning",
    "red_flag",
)


def _normalize_text(t: Any) -> str:
    if not isinstance(t, str):
        return ""
    return t.strip()


def _is_meaningful(s: str, *, min_len: int = 2) -> bool:
    return bool(s) and len(s) >= min_len


def _copy_citations(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = item.get("citations") if isinstance(item, dict) else None
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for c in raw:
        if isinstance(c, dict):
            out.append(dict(c))
    return out


def extract_claims(
    answer_json: Optional[Dict[str, Any]],
    *,
    case_id: str,
    mode: str,
    max_claims: int = 20,
) -> List[Dict[str, Any]]:
    """Достаёт plain-список claims с проверяемыми текстами.

    Порядок приоритета (для clamp до max_claims):
        diagnosis → recommendation → red_flag → reasoning →
        rec_reasoning → finding.

    `citations` копируются из ближайшего родительского item.
    `source_item_path` — для трейсинга ('differential_diagnoses[0].reasoning').
    """
    if not isinstance(answer_json, dict):
        return []

    diagnoses = answer_json.get("differential_diagnoses") or []
    next_steps = answer_json.get("recommended_next_steps") or []
    red_flags = answer_json.get("red_flags") or []

    buckets: Dict[str, List[Dict[str, Any]]] = {t: [] for t in CLAIM_TYPES}

    
    for i, d in enumerate(diagnoses):
        if not isinstance(d, dict):
            continue
        cits = _copy_citations(d)
        diag_text = _normalize_text(d.get("diagnosis"))
        if _is_meaningful(diag_text):
            buckets["diagnosis"].append(
                _claim_record(case_id, mode, "diagnosis", diag_text, cits,
                              f"differential_diagnoses[{i}].diagnosis")
            )
        reasoning_text = _normalize_text(d.get("reasoning"))
        if _is_meaningful(reasoning_text, min_len=10):
            buckets["reasoning"].append(
                _claim_record(case_id, mode, "reasoning", reasoning_text, cits,
                              f"differential_diagnoses[{i}].reasoning")
            )
        for j, sf in enumerate(d.get("supporting_findings") or []):
            sf_text = _normalize_text(sf)
            if _is_meaningful(sf_text):
                buckets["finding"].append(
                    _claim_record(case_id, mode, "finding", sf_text, cits,
                                  f"differential_diagnoses[{i}].supporting_findings[{j}]")
                )

    
    for i, r in enumerate(next_steps):
        if not isinstance(r, dict):
            continue
        cits = _copy_citations(r)
        rec_text = _normalize_text(r.get("recommendation"))
        if _is_meaningful(rec_text):
            buckets["recommendation"].append(
                _claim_record(case_id, mode, "recommendation", rec_text, cits,
                              f"recommended_next_steps[{i}].recommendation")
            )
        rrt = _normalize_text(r.get("reasoning"))
        if _is_meaningful(rrt, min_len=10):
            buckets["rec_reasoning"].append(
                _claim_record(case_id, mode, "rec_reasoning", rrt, cits,
                              f"recommended_next_steps[{i}].reasoning")
            )

    
    for i, rf in enumerate(red_flags):
        if not isinstance(rf, dict):
            continue
        cits = _copy_citations(rf)
        rf_text = _normalize_text(rf.get("red_flag"))
        if _is_meaningful(rf_text):
            buckets["red_flag"].append(
                _claim_record(case_id, mode, "red_flag", rf_text, cits,
                              f"red_flags[{i}].red_flag")
            )

    priority = ("diagnosis", "recommendation", "red_flag", "reasoning",
                "rec_reasoning", "finding")
    ordered: List[Dict[str, Any]] = []
    for k in priority:
        ordered.extend(buckets.get(k, []))

    
    out: List[Dict[str, Any]] = []
    for n, c in enumerate(ordered[:max_claims], start=1):
        c["claim_id"] = f"{mode}_{case_id}_c{n:03d}"
        out.append(c)
    return out


def _claim_record(
    case_id: str,
    mode: str,
    claim_type: str,
    text: str,
    citations: List[Dict[str, Any]],
    source_item_path: str,
) -> Dict[str, Any]:
    return {
        "claim_id": "",  
        "case_id": case_id,
        "mode": mode,
        "claim_type": claim_type,
        "claim_text": text,
        "citations": citations,
        "source_item_path": source_item_path,
    }






def format_reference_context(
    retrieved_chunks: Sequence[Dict[str, Any]],
    *,
    text_chars_per_chunk: int = 1500,
) -> str:
    """Форматирует retrieved_chunks (записи из rag.retrieved_chunks) для judge.

    На входе — записи `hit_to_chunk_record`-формата (см. rag_generation.py),
    на выходе — текст с блоками [Источник S1]..[Источник SN] (тот же layout,
    что в format_retrieved_context, чтобы judge видел одинаковый формат).
    """
    blocks: List[str] = []
    for i, c in enumerate(retrieved_chunks or [], start=1):
        if not isinstance(c, dict):
            continue
        sid = c.get("source_id") or f"S{i}"
        chunk_id = c.get("chunk_id") or ""
        document_id = c.get("document_id") or ""
        document_title = c.get("document_title") or ""
        section_title = c.get("section_title") or ""
        label = c.get("label") or ""
        ps, pe = c.get("page_start"), c.get("page_end")
        if ps is not None and pe is not None:
            pages = f"{ps}-{pe}"
        elif ps is not None:
            pages = str(ps)
        else:
            pages = ""
        text = (c.get("text") or "").strip()
        if len(text) > text_chars_per_chunk:
            text = text[:text_chars_per_chunk] + "…"
        blocks.append(
            f"[Источник {sid}]\n"
            f"chunk_id: {chunk_id}\n"
            f"document_id: {document_id}\n"
            f"document_title: {document_title}\n"
            f"section_title: {section_title}\n"
            f"label: {label}\n"
            f"pages: {pages}\n"
            f"text:\n{text}\n"
        )
    return "\n".join(blocks)


def format_claims_for_judge(claims: Sequence[Dict[str, Any]]) -> str:
    """Компактный JSON со списком claims для judge.

    Намеренно отдаём только то, что judge должен видеть и сохранять
    обратно: claim_id, claim_type, claim_text, citations.
    """
    payload = [
        {
            "claim_id": c.get("claim_id"),
            "claim_type": c.get("claim_type"),
            "claim_text": c.get("claim_text"),
            "citations": c.get("citations") or [],
        }
        for c in claims
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_citations_for_judge(
    claims: Sequence[Dict[str, Any]],
    retrieved_chunks: Sequence[Dict[str, Any]],
    *,
    text_chars_per_block: int = 800,
) -> str:
    """Формирует плоский список citation-объектов для citation-accuracy judge.

    Для каждой citation в каждом claim делает строку:
        {citation_id, claim_id, claim_text, citation, source_block_text}
    где source_block_text — содержимое соответствующего [Источник SN]
    или null/пусто, если source_id citation отсутствует в retrieved_chunks.
    """
    chunks_by_sid: Dict[str, Dict[str, Any]] = {}
    for c in retrieved_chunks or []:
        if isinstance(c, dict) and c.get("source_id"):
            chunks_by_sid[str(c["source_id"])] = c

    out: List[Dict[str, Any]] = []
    counter = 0
    for cl in claims:
        for cit in cl.get("citations") or []:
            if not isinstance(cit, dict):
                continue
            counter += 1
            sid = str(cit.get("source_id") or "")
            ref = chunks_by_sid.get(sid)
            if ref:
                src_text = (ref.get("text") or "").strip()
                if len(src_text) > text_chars_per_block:
                    src_text = src_text[:text_chars_per_block] + "…"
                source_block_text = (
                    f"[Источник {sid}]\n"
                    f"chunk_id: {ref.get('chunk_id', '')}\n"
                    f"document_id: {ref.get('document_id', '')}\n"
                    f"document_title: {ref.get('document_title', '')}\n"
                    f"section_title: {ref.get('section_title', '')}\n"
                    f"label: {ref.get('label', '')}\n"
                    f"pages: {ref.get('page_start') or ''}-{ref.get('page_end') or ''}\n"
                    f"text:\n{src_text}\n"
                )
            else:
                source_block_text = (
                    f"[Источник {sid}] — НЕ НАЙДЕН в reference_context. "
                    f"Это автоматически означает is_valid_reference=false."
                )
            out.append(
                {
                    "citation_id": f"{cl.get('claim_id')}_cit{counter:03d}",
                    "claim_id": cl.get("claim_id"),
                    "claim_text": cl.get("claim_text"),
                    "citation": cit,
                    "source_block_text": source_block_text,
                }
            )
    return json.dumps(out, ensure_ascii=False, indent=2)






_JUDGE_RETRY_HINT = (
    "\n\nЗАМЕЧАНИЕ: предыдущий ответ не был валидным JSON по схеме. "
    "ВЕРНИ строго один валидный JSON-объект по указанной схеме. Никакого "
    "текста до или после JSON, никаких markdown-блоков."
)


class JudgeRunner:
    """Generic-обёртка над LLMClient для judge-вызовов.

    Один JudgeRunner = один judge-config = один тип проверки.
    """

    def __init__(
        self,
        client: LLMClient,
        prompt_config: Dict[str, Any],
        *,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.prompt_config = prompt_config
        self.max_retries = int(max_retries)
        self.judge_name = str(prompt_config.get("name") or "judge")

    def run(self, **placeholders: Any) -> Dict[str, Any]:
        """Делает 1 judge-вызов и парсит JSON.

        Возвращает dict с ключами:
            parsed:    Optional[dict]   — распарсенный JSON или None
            raw_text:  str              — сырой текст ответа LLM
            errors:    List[str]
            elapsed_sec: float
            usage:     Optional[dict]
            mock:      bool
            attempts:  int
        """
        
        if self.client.provider_type == "mock":
            t0 = time.perf_counter()
            parsed = _mock_judge_payload(self.judge_name, placeholders)
            elapsed = time.perf_counter() - t0
            raw_text = json.dumps(parsed, ensure_ascii=False)
            return {
                "parsed": parsed,
                "raw_text": raw_text,
                "errors": [],
                "elapsed_sec": round(elapsed, 4),
                "usage": None,
                "mock": True,
                "attempts": 1,
            }

        messages = build_messages_from_judge_prompt_config(
            self.prompt_config, **placeholders
        )

        errors: List[str] = []
        attempts = 0
        last_text = ""
        last_usage: Optional[Dict[str, Any]] = None
        last_elapsed = 0.0

        for attempt in range(self.max_retries + 1):
            attempts += 1
            try:
                resp = self.client.generate(messages)
            except Exception as e:  
                errors.append(f"llm_call_failed: {e}")
                logger.exception("Judge %s llm call failed (attempt %d)", self.judge_name, attempts)
                return {
                    "parsed": None,
                    "raw_text": last_text,
                    "errors": errors,
                    "elapsed_sec": round(last_elapsed, 4),
                    "usage": last_usage,
                    "mock": False,
                    "attempts": attempts,
                }

            last_text = resp.get("text", "") or ""
            last_usage = resp.get("usage")
            last_elapsed += float(resp.get("elapsed_sec") or 0.0)

            parsed, perrs = parse_llm_json(last_text)
            if parsed is not None:
                return {
                    "parsed": parsed,
                    "raw_text": last_text,
                    "errors": errors + perrs,
                    "elapsed_sec": round(last_elapsed, 4),
                    "usage": last_usage,
                    "mock": False,
                    "attempts": attempts,
                }

            errors.extend([f"attempt{attempt+1}: {e}" for e in perrs])
            if attempt < self.max_retries:
                
                hint_msg = {
                    "role": "user",
                    "content": (
                        f"Сырой ответ был:\n```\n{last_text[:1500]}\n```"
                        + _JUDGE_RETRY_HINT
                    ),
                }
                messages = list(messages) + [hint_msg]

        return {
            "parsed": None,
            "raw_text": last_text,
            "errors": errors,
            "elapsed_sec": round(last_elapsed, 4),
            "usage": last_usage,
            "mock": False,
            "attempts": attempts,
        }






def run_faithfulness_judge(
    runner: JudgeRunner,
    *,
    case_id: str,
    mode: str,
    patient_case: str,
    claims: Sequence[Dict[str, Any]],
    reference_context: str,
) -> Dict[str, Any]:
    return runner.run(
        case_id=case_id,
        mode=mode,
        patient_case=patient_case,
        reference_context=reference_context,
        claims_json=format_claims_for_judge(claims),
    )


def run_citation_accuracy_judge(
    runner: JudgeRunner,
    *,
    case_id: str,
    patient_case: str,
    claims: Sequence[Dict[str, Any]],
    retrieved_chunks: Sequence[Dict[str, Any]],
    reference_context: Optional[str] = None,
) -> Dict[str, Any]:
    if reference_context is None:
        reference_context = format_reference_context(retrieved_chunks)
    return runner.run(
        case_id=case_id,
        patient_case=patient_case,
        reference_context=reference_context,
        citations_json=format_citations_for_judge(claims, retrieved_chunks),
    )


def run_answer_relevance_judge(
    runner: JudgeRunner,
    *,
    case_id: str,
    mode: str,
    patient_case: str,
    answer_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    answer_json_str = json.dumps(answer_json or {}, ensure_ascii=False, indent=2)
    return runner.run(
        case_id=case_id,
        mode=mode,
        patient_case=patient_case,
        answer_json_str=answer_json_str,
    )






def _mock_judge_payload(judge_name: str, placeholders: Dict[str, Any]) -> Dict[str, Any]:
    
    name = (judge_name or "").lower()
    case_id = str(placeholders.get("case_id") or "mock_case")
    mode = str(placeholders.get("mode") or "rag")

    if "faithfulness" in name:
        
        claims_raw = placeholders.get("claims_json") or "[]"
        try:
            claims = json.loads(claims_raw) if isinstance(claims_raw, str) else (claims_raw or [])
        except Exception:  
            claims = []
        evals = []
        for c in claims:
            cits = c.get("citations") or []
            has_cit = bool(cits)
            sids = sorted({str(x.get("source_id")) for x in cits if isinstance(x, dict) and x.get("source_id")})
            evals.append({
                "claim_id": c.get("claim_id"),
                "claim_text": c.get("claim_text", ""),
                "support_status": "supported",
                "support_score": 0.95,
                "citation_supports_claim": True if (mode == "rag" and has_cit) else None,
                "uses_valid_citation": True if (mode == "rag" and has_cit) else None,
                "supporting_source_ids": sids or (["S1"] if mode == "rag" else []),
                "explanation": "MOCK: claim считается supported для smoke-теста.",
                "problems": [],
            })
        n = max(1, len(evals))
        return {
            "case_id": case_id,
            "mode": mode,
            "claim_evaluations": evals,
            "overall": {
                "faithfulness_score": 1.0,
                "hallucination_rate": 0.0,
                "summary": "MOCK overall: all claims supported (smoke).",
            },
        }

    if "citation_accuracy" in name or "citation-accuracy" in name:
        cits_raw = placeholders.get("citations_json") or "[]"
        try:
            cits = json.loads(cits_raw) if isinstance(cits_raw, str) else (cits_raw or [])
        except Exception:  
            cits = []
        evals = []
        for c in cits:
            sb = (c.get("source_block_text") or "")
            valid = "НЕ НАЙДЕН" not in sb
            evals.append({
                "citation_id": c.get("citation_id"),
                "claim_id": c.get("claim_id"),
                "is_valid_reference": valid,
                "supports_claim": valid,
                "explanation": "MOCK: citation считается валидной/поддерживающей." if valid else "MOCK: source не найден в reference_context.",
            })
        return {
            "case_id": case_id,
            "citation_evaluations": evals,
        }

    if "answer_relevance" in name or "relevance" in name:
        return {
            "case_id": case_id,
            "mode": mode,
            "answer_relevance_score": 4,
            "clinical_usefulness_score": 4,
            "safety_score": 5,
            "has_disclaimer": True,
            "states_final_diagnosis": False,
            "summary": "MOCK: ответ считается релевантным и безопасным.",
        }

    
    return {"mock_judge_unknown_name": judge_name}
