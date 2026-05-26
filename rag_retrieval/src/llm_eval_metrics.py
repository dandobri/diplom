"""Агрегация метрик LLM evaluation.

Чистый Python, без вызовов LLM. На вход получает claim/case-evaluations
от judgeев и считает per-case + global метрики, нужные для ВКР.

Главные термины:
    - faithfulness_strict = supported / total_claims
    - faithfulness_soft   = (supported + 0.5*partially_supported) / total
    - hallucination_rate  = (unsupported + contradicted) / total_claims
    - citation_validity_rate (technical) — берётся из validate_citations
    - citation_coverage_item_level — % diagnosis/recommendation items с >= 1 citation
    - citation_coverage_claim_level — % claims с >= 1 citation
    - citation_accuracy_rate — % citations, поддержанных judge-ом

Цели ВКР (зашиты как константы):
    TARGET_FAITHFULNESS_RAG = 0.85
    TARGET_HALLUCINATION_RAG = 0.15
    TARGET_CITATION_ACCURACY = 0.90
    TARGET_FAITHFULNESS_IMPROVEMENT_ABS = 0.20
    TARGET_HALLUCINATION_REDUCTION_ABS = 0.20  (proxy: «снижение недостоверных <15%»)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from .rag_generation import validate_citations

logger = logging.getLogger(__name__)


TARGET_FAITHFULNESS_RAG: float = 0.85
TARGET_HALLUCINATION_RAG: float = 0.15
TARGET_CITATION_ACCURACY: float = 0.90
TARGET_FAITHFULNESS_IMPROVEMENT_ABS: float = 0.20
TARGET_HALLUCINATION_REDUCTION_ABS: float = 0.20

SUPPORT_STATUSES = (
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "not_enough_information",
)


def _round(v: Optional[float], n: int = 4) -> Optional[float]:
    return round(v, n) if isinstance(v, (int, float)) else None


def _avg(values: Sequence[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    return float(sum(cleaned) / len(cleaned)) if cleaned else None


def _safe_divide(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def aggregate_per_case(
    *,
    case_id: str,
    mode: str,
    claim_evals: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    
    counts: Dict[str, int] = {s: 0 for s in SUPPORT_STATUSES}
    cit_supports_yes = 0
    cit_supports_no = 0
    cit_supports_total = 0
    valid_cit_yes = 0
    valid_cit_no = 0

    for ce in claim_evals or []:
        s = (ce.get("support_status") or "").strip()
        if s in counts:
            counts[s] += 1
        csc = ce.get("citation_supports_claim")
        if isinstance(csc, bool):
            cit_supports_total += 1
            if csc:
                cit_supports_yes += 1
            else:
                cit_supports_no += 1
        uvc = ce.get("uses_valid_citation")
        if isinstance(uvc, bool):
            if uvc:
                valid_cit_yes += 1
            else:
                valid_cit_no += 1

    total = sum(counts.values())
    supported = counts["supported"]
    partial = counts["partially_supported"]
    unsupported = counts["unsupported"]
    contradicted = counts["contradicted"]
    nei = counts["not_enough_information"]

    faithfulness_strict = _safe_divide(supported, total)
    faithfulness_soft = _safe_divide(supported + 0.5 * partial, total)
    hallucination_rate = _safe_divide(unsupported + contradicted, total)
    citation_supports_rate = _safe_divide(cit_supports_yes, cit_supports_total)

    return {
        "case_id": case_id,
        "mode": mode,
        "num_claims": total,
        "supported_claims": supported,
        "partially_supported_claims": partial,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
        "not_enough_information_claims": nei,
        "faithfulness_strict": _round(faithfulness_strict),
        "faithfulness_soft": _round(faithfulness_soft),
        "hallucination_rate": _round(hallucination_rate),
        "citation_supports_rate_from_faithfulness": _round(citation_supports_rate),
        "citation_supports_yes": cit_supports_yes,
        "citation_supports_no": cit_supports_no,
        "citation_supports_total": cit_supports_total,
        "uses_valid_citation_yes": valid_cit_yes,
        "uses_valid_citation_no": valid_cit_no,
    }


def compute_citation_metrics_for_rag(
    *,
    answer_json: Optional[Dict[str, Any]],
    retrieved_chunks: Sequence[Dict[str, Any]],
    claims: Sequence[Dict[str, Any]],
    claim_evals: Sequence[Dict[str, Any]],
    citation_evals: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Считает все citation-метрики для одного RAG-ответа.

    `citation_validity_rate` — техническая (через validate_citations).
    `citation_coverage_item_level` — берём из validate_citations (старая
        метрика по diagnosis/recommendation items).
    `citation_coverage_claim_level` — claims с >= 1 citation / total claims.
    `citation_accuracy_rate` — citation_evals.supports_claim==True / total
        (если citation-judge запускался). Иначе fallback на
        claim_evals.citation_supports_claim==True / total.
    """
    cv = validate_citations(answer_json, list(retrieved_chunks))

    n_total_claims = len(claims) or 0
    n_claims_with_cit = sum(1 for c in claims if c.get("citations"))
    coverage_claim = _safe_divide(n_claims_with_cit, n_total_claims)

    if citation_evals:
        n_pos = 0
        n_total = 0
        for ce in citation_evals:
            sup = ce.get("supports_claim")
            if isinstance(sup, bool):
                n_total += 1
                if sup:
                    n_pos += 1
        citation_accuracy_rate = _safe_divide(n_pos, n_total)
        accuracy_source = "citation_judge"
    else:
        n_pos = 0
        n_total = 0
        for ce in claim_evals:
            csc = ce.get("citation_supports_claim")
            if isinstance(csc, bool):
                n_total += 1
                if csc:
                    n_pos += 1
        citation_accuracy_rate = _safe_divide(n_pos, n_total)
        accuracy_source = "faithfulness_judge_fallback"

    total_cit = int(cv.get("citation_count") or 0)
    invalid_cit = int(cv.get("invalid_citation_count") or 0)
    citation_validity_rate = (
        (total_cit - invalid_cit) / total_cit if total_cit > 0 else None
    )

    return {
        "citation_count": total_cit,
        "invalid_citation_count": invalid_cit,
        "citation_validity_rate": _round(citation_validity_rate),
        "citation_coverage_item_level": cv.get("citation_coverage_estimate"),
        "citation_coverage_claim_level": _round(coverage_claim),
        "citation_accuracy_rate": _round(citation_accuracy_rate),
        "citation_accuracy_source": accuracy_source,
    }




_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _normalize_diag(s: str) -> str:
    return s.casefold().strip() if isinstance(s, str) else ""


def _tokens(s: str) -> set:
    return set(_TOKEN_RE.findall(_normalize_diag(s)))


def _diag_match(expected: str, candidate: str, *, jaccard_threshold: float = 0.5) -> bool:
    
    e = _normalize_diag(expected)
    c = _normalize_diag(candidate)
    if not e or not c:
        return False
    if e in c or c in e:
        return True
    et, ct = _tokens(e), _tokens(c)
    if not et or not ct:
        return False
    inter = len(et & ct)
    union = len(et | ct)
    return (inter / union) >= jaccard_threshold


def match_diagnoses(
    expected: Sequence[str],
    answer_json: Optional[Dict[str, Any]],
    *,
    top1_k: int = 1,
    top3_k: int = 3,
) -> Optional[Dict[str, Any]]:
    """Считает top1 / top3 / coverage match по expected_diagnoses.

    Возвращает None, если expected пусто (метрики неприменимы).
    """
    if not expected:
        return None
    if not isinstance(answer_json, dict):
        return {
            "expected": list(expected),
            "predicted": [],
            "top1_match": False,
            "top3_match": False,
            "diagnosis_coverage": 0.0,
        }
    diagnoses = answer_json.get("differential_diagnoses") or []
    predicted: List[str] = []
    for d in diagnoses:
        if isinstance(d, dict):
            t = d.get("diagnosis")
            if isinstance(t, str) and t.strip():
                predicted.append(t.strip())

    top1 = any(_diag_match(e, p) for e in expected for p in predicted[:top1_k])
    top3 = any(_diag_match(e, p) for e in expected for p in predicted[:top3_k])

    matched = sum(1 for e in expected if any(_diag_match(e, p) for p in predicted))
    coverage = matched / len(expected) if expected else 0.0

    return {
        "expected": list(expected),
        "predicted": predicted,
        "top1_match": bool(top1),
        "top3_match": bool(top3),
        "diagnosis_coverage": round(coverage, 4),
    }




def aggregate_global(case_evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Свёртка per-case → global RAG/no-RAG + comparison block + targets.

    Каждое case_evaluation должно содержать ключи:
        case_id, mode,
        per_case (от aggregate_per_case),
        citation_metrics (Optional, только для RAG; от compute_citation_metrics_for_rag),
        relevance (Optional; от answer_relevance_judge.parsed),
        valid_json (bool),
        diagnosis_match (Optional).
    """
    rag_cases = [c for c in case_evaluations if c.get("mode") == "rag"]
    no_rag_cases = [c for c in case_evaluations if c.get("mode") == "no_rag"]

    rag_block = _aggregate_mode_block(rag_cases, mode="rag")
    no_rag_block = _aggregate_mode_block(no_rag_cases, mode="no_rag")

    
    cmp_block: Dict[str, Any] = {}
    f_rag = rag_block.get("faithfulness_soft")
    f_no = no_rag_block.get("faithfulness_soft")
    if isinstance(f_rag, (int, float)) and isinstance(f_no, (int, float)):
        cmp_block["faithfulness_soft_improvement_abs"] = _round(f_rag - f_no)
        cmp_block["faithfulness_soft_improvement_rel"] = (
            _round((f_rag - f_no) / f_no) if f_no else None
        )
    h_rag = rag_block.get("hallucination_rate")
    h_no = no_rag_block.get("hallucination_rate")
    if isinstance(h_rag, (int, float)) and isinstance(h_no, (int, float)):
        cmp_block["hallucination_rate_reduction_abs"] = _round(h_no - h_rag)

    rag_meets_faithfulness = _ge(rag_block.get("faithfulness_soft"), TARGET_FAITHFULNESS_RAG)
    rag_meets_hallucination = _le_strict(rag_block.get("hallucination_rate"), TARGET_HALLUCINATION_RAG)
    rag_meets_citation = _ge(rag_block.get("citation_accuracy_rate"), TARGET_CITATION_ACCURACY)
    rag_meets_improvement = _ge(
        cmp_block.get("faithfulness_soft_improvement_abs"),
        TARGET_FAITHFULNESS_IMPROVEMENT_ABS,
    )
    rag_meets_hallucination_reduction = _ge(
        cmp_block.get("hallucination_rate_reduction_abs"),
        TARGET_HALLUCINATION_REDUCTION_ABS,
    )

    cmp_block.update({
        "rag_meets_faithfulness_target": rag_meets_faithfulness,
        "rag_meets_hallucination_target": rag_meets_hallucination,
        "rag_meets_citation_accuracy_target": rag_meets_citation,
        "rag_meets_faithfulness_improvement_target": rag_meets_improvement,
        "rag_meets_hallucination_reduction_target": rag_meets_hallucination_reduction,
    })
    wins = losses = ties = 0
    for r in rag_cases:
        cid = r.get("case_id")
        match = next((c for c in no_rag_cases if c.get("case_id") == cid), None)
        if not match:
            continue
        rf = (r.get("per_case") or {}).get("faithfulness_soft")
        nf = (match.get("per_case") or {}).get("faithfulness_soft")
        if not isinstance(rf, (int, float)) or not isinstance(nf, (int, float)):
            continue
        if rf > nf:
            wins += 1
        elif rf < nf:
            losses += 1
        else:
            ties += 1
    cmp_block["case_level_wins_rag_better"] = wins
    cmp_block["case_level_losses_rag_worse"] = losses
    cmp_block["case_level_ties"] = ties

    return {
        "num_cases": max(len(rag_cases), len(no_rag_cases)),
        "rag": rag_block,
        "no_rag": no_rag_block,
        "comparison": cmp_block,
        "targets": {
            "TARGET_FAITHFULNESS_RAG": TARGET_FAITHFULNESS_RAG,
            "TARGET_HALLUCINATION_RAG": TARGET_HALLUCINATION_RAG,
            "TARGET_CITATION_ACCURACY": TARGET_CITATION_ACCURACY,
            "TARGET_FAITHFULNESS_IMPROVEMENT_ABS": TARGET_FAITHFULNESS_IMPROVEMENT_ABS,
            "TARGET_HALLUCINATION_REDUCTION_ABS": TARGET_HALLUCINATION_REDUCTION_ABS,
        },
    }


def _aggregate_mode_block(
    mode_cases: Sequence[Dict[str, Any]],
    *,
    mode: str,
) -> Dict[str, Any]:
    n = len(mode_cases)
    if not n:
        return {"mode": mode, "num_cases": 0}

    valid_json = sum(1 for c in mode_cases if c.get("valid_json"))
    failed_cases = [c.get("case_id") for c in mode_cases if not c.get("valid_json")]

    pcs = [c.get("per_case") or {} for c in mode_cases]
    f_strict = _avg([p.get("faithfulness_strict") for p in pcs])
    f_soft = _avg([p.get("faithfulness_soft") for p in pcs])
    h_rate = _avg([p.get("hallucination_rate") for p in pcs])

    
    judged_cases = sum(
        1 for p in pcs if isinstance(p.get("faithfulness_soft"), (int, float))
    )
    failed_judge_cases = [
        c.get("case_id")
        for c, p in zip(mode_cases, pcs)
        if not isinstance(p.get("faithfulness_soft"), (int, float))
    ]

    total_claims = sum(int(p.get("num_claims") or 0) for p in pcs)
    total_supported = sum(int(p.get("supported_claims") or 0) for p in pcs)
    total_partial = sum(int(p.get("partially_supported_claims") or 0) for p in pcs)
    total_unsup = sum(int(p.get("unsupported_claims") or 0) for p in pcs)
    total_contra = sum(int(p.get("contradicted_claims") or 0) for p in pcs)
    total_nei = sum(int(p.get("not_enough_information_claims") or 0) for p in pcs)

    block: Dict[str, Any] = {
        "mode": mode,
        "num_cases": n,
        "valid_json_count": valid_json,
        "valid_json_rate": _round(valid_json / n) if n else None,
        "failed_cases": failed_cases,
        "judged_cases": judged_cases,                      
        "judge_failed_cases": failed_judge_cases,          
        "judge_success_rate": _round(judged_cases / n) if n else None,
        "total_claims": total_claims,
        "total_supported_claims": total_supported,
        "total_partially_supported_claims": total_partial,
        "total_unsupported_claims": total_unsup,
        "total_contradicted_claims": total_contra,
        "total_not_enough_information_claims": total_nei,
        
        "faithfulness_strict": _round(f_strict),
        "faithfulness_soft": _round(f_soft),
        "hallucination_rate": _round(h_rate),
        
        "faithfulness_strict_micro": _round(_safe_divide(total_supported, total_claims)),
        "faithfulness_soft_micro": _round(_safe_divide(total_supported + 0.5 * total_partial, total_claims)),
        "hallucination_rate_micro": _round(_safe_divide(total_unsup + total_contra, total_claims)),
    }

    if mode == "rag":
        cms = [c.get("citation_metrics") or {} for c in mode_cases]
        cov_item = _avg([m.get("citation_coverage_item_level") for m in cms])
        cov_claim = _avg([m.get("citation_coverage_claim_level") for m in cms])
        validity = _avg([m.get("citation_validity_rate") for m in cms])
        accuracy = _avg([m.get("citation_accuracy_rate") for m in cms])
        invalid_total = sum(int(m.get("invalid_citation_count") or 0) for m in cms)
        block.update({
            "citation_coverage_item_level": _round(cov_item),
            "citation_coverage_claim_level": _round(cov_claim),
            "citation_validity_rate": _round(validity),
            "citation_accuracy_rate": _round(accuracy),
            "invalid_citation_count_total": invalid_total,
        })

    
    rels = [c.get("relevance") or {} for c in mode_cases]
    if any(rels):
        block.update({
            "answer_relevance_avg": _round(_avg([r.get("answer_relevance_score") for r in rels])),
            "clinical_usefulness_avg": _round(_avg([r.get("clinical_usefulness_score") for r in rels])),
            "safety_avg": _round(_avg([r.get("safety_score") for r in rels])),
            "states_final_diagnosis_count": sum(
                1 for r in rels if r.get("states_final_diagnosis") is True
            ),
            "has_disclaimer_count": sum(
                1 for r in rels if r.get("has_disclaimer") is True
            ),
        })

    
    dms = [c.get("diagnosis_match") for c in mode_cases]
    dms_present = [d for d in dms if isinstance(d, dict)]
    if dms_present:
        block.update({
            "top1_diagnosis_match_rate": _round(
                sum(1 for d in dms_present if d.get("top1_match")) / len(dms_present)
            ),
            "top3_diagnosis_match_rate": _round(
                sum(1 for d in dms_present if d.get("top3_match")) / len(dms_present)
            ),
            "diagnosis_coverage_avg": _round(_avg([d.get("diagnosis_coverage") for d in dms_present])),
            "diagnosis_match_evaluated_cases": len(dms_present),
        })

    return block


def _ge(value: Any, threshold: float) -> Optional[bool]:
    if not isinstance(value, (int, float)):
        return None
    return value >= threshold


def _le_strict(value: Any, threshold: float) -> Optional[bool]:
    if not isinstance(value, (int, float)):
        return None
    return value < threshold
