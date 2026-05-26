from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io_utils import (
    ensure_dir,
    load_chunks_any,
    setup_logging,
    write_jsonl,
)

logger = logging.getLogger(__name__)

QUERY_TYPE_DISTRIBUTION: Dict[str, float] = {
    "treatment": 0.22,
    "diagnosis": 0.22,
    "symptoms": 0.18,
    "lab_tests": 0.08,
    "instrumental_tests": 0.07,
    "risk_factors": 0.06,
    "contraindications": 0.06,
    "follow_up": 0.06,
    "differential_diagnosis": 0.03,
    "emergency": 0.01,
    "general": 0.01,
}

DIFFICULTY_DISTRIBUTION: Dict[str, float] = {
    "easy": 0.30,
    "medium": 0.50,
    "hard": 0.20,
}

LABEL_TO_QUERY_TYPE: Dict[str, str] = {
    "treatment": "treatment",
    "therapy": "treatment",
    "лечение": "treatment",
    "diagnosis": "diagnosis",
    "диагностика": "diagnosis",
    "differential_diagnosis": "differential_diagnosis",
    "symptoms": "symptoms",
    "clinical_picture": "symptoms",
    "клиническая_картина": "symptoms",
    "lab_tests": "lab_tests",
    "laboratory": "lab_tests",
    "instrumental_tests": "instrumental_tests",
    "instrumental": "instrumental_tests",
    "imaging": "instrumental_tests",
    "risk_factors": "risk_factors",
    "etiology": "risk_factors",
    "contraindications": "contraindications",
    "follow_up": "follow_up",
    "monitoring": "follow_up",
    "rehabilitation": "follow_up",
    "emergency": "emergency",
    "urgent": "emergency",
}

TEMPLATES: Dict[Tuple[str, str], List[str]] = {
    ("treatment", "easy"): [
        "лечение {topic}",
        "как лечить {topic}",
        "терапия {topic}",
    ],
    ("treatment", "medium"): [
        "какое лечение рекомендуется при {topic}",
        "методы терапии {topic}",
        "тактика лечения пациентов с {topic}",
    ],
    ("treatment", "hard"): [
        "чем лечат пациентов с {topic}",
        "что делать если у пациента {topic}",
        "пациент с {topic} какие назначения",
    ],
    ("diagnosis", "easy"): [
        "диагностика {topic}",
        "как диагностировать {topic}",
        "критерии {topic}",
    ],
    ("diagnosis", "medium"): [
        "критерии диагностики {topic}",
        "на основании чего ставится диагноз {topic}",
        "как подтвердить диагноз {topic}",
    ],
    ("diagnosis", "hard"): [
        "когда нужно подозревать {topic}",
        "какие тесты подтверждают {topic}",
        "что является основанием для диагноза {topic}",
    ],
    ("differential_diagnosis", "easy"): [
        "дифференциальная диагностика {topic}",
        "с чем дифференцировать {topic}",
    ],
    ("differential_diagnosis", "medium"): [
        "от каких заболеваний нужно отличать {topic}",
        "дифф диагноз {topic}",
    ],
    ("differential_diagnosis", "hard"): [
        "пациент с симптомами {topic} какие еще заболевания исключить",
    ],
    ("symptoms", "easy"): [
        "симптомы {topic}",
        "клинические проявления {topic}",
    ],
    ("symptoms", "medium"): [
        "какие симптомы характерны для {topic}",
        "признаки {topic}",
    ],
    ("symptoms", "hard"): [
        "у пациента {topic} на что обратить внимание",
        "какие жалобы предъявляют пациенты с {topic}",
    ],
    ("lab_tests", "easy"): [
        "лабораторная диагностика {topic}",
        "какие анализы при {topic}",
    ],
    ("lab_tests", "medium"): [
        "какие лабораторные исследования назначают при {topic}",
        "лабораторные критерии {topic}",
    ],
    ("lab_tests", "hard"): [
        "какие показатели крови меняются при {topic}",
        "какие анализы нужны для подтверждения {topic}",
    ],
    ("instrumental_tests", "easy"): [
        "инструментальная диагностика {topic}",
        "какие обследования при {topic}",
    ],
    ("instrumental_tests", "medium"): [
        "какие инструментальные методы используют при {topic}",
        "визуализация при {topic}",
    ],
    ("instrumental_tests", "hard"): [
        "какое исследование выбрать пациенту с подозрением на {topic}",
    ],
    ("risk_factors", "easy"): [
        "факторы риска {topic}",
        "причины {topic}",
    ],
    ("risk_factors", "medium"): [
        "что повышает риск развития {topic}",
        "этиология {topic}",
    ],
    ("risk_factors", "hard"): [
        "у каких пациентов чаще встречается {topic}",
    ],
    ("contraindications", "easy"): [
        "противопоказания {topic}",
        "когда нельзя {topic}",
    ],
    ("contraindications", "medium"): [
        "противопоказания при лечении {topic}",
        "ограничения для пациентов с {topic}",
    ],
    ("contraindications", "hard"): [
        "когда нельзя назначать терапию при {topic}",
    ],
    ("follow_up", "easy"): [
        "наблюдение пациентов с {topic}",
        "диспансерное наблюдение {topic}",
    ],
    ("follow_up", "medium"): [
        "как наблюдать пациента с {topic}",
        "контроль состояния при {topic}",
    ],
    ("follow_up", "hard"): [
        "как часто обследовать пациента с {topic}",
    ],
    ("emergency", "easy"): [
        "неотложная помощь {topic}",
        "экстренная помощь при {topic}",
    ],
    ("emergency", "medium"): [
        "что делать при остром {topic}",
        "неотложные мероприятия при {topic}",
    ],
    ("emergency", "hard"): [
        "пациент с {topic} в тяжелом состоянии что делать",
    ],
    ("general", "easy"): [
        "клинические рекомендации {topic}",
        "что важно знать о {topic}",
    ],
    ("general", "medium"): [
        "обзор клинических рекомендаций по {topic}",
    ],
    ("general", "hard"): [
        "что включают клинические рекомендации по {topic}",
    ],
}


JUNK_SECTION_KEYWORDS = (
    "литератур",
    "источник",
    "оглавлен",
    "содержан",
    "приложен",
    "состав рабоч",
    "критерии оцен",
    "термины и опре",
    "список сокращ",
)
JUNK_LABELS = {"references", "toc", "appendix", "abbreviations", "glossary"}

RU_STOPWORDS = {
    "при", "для", "это", "что", "как", "или", "также", "более", "менее",
    "лечение", "диагностика", "симптомы",
    "подход", "метод", "методы", "общий", "основной", "основные",
    "пациент", "пациенты", "пациента", "пациентов",
    "the", "and", "with", "for", "from",
}


@dataclass
class Candidate:
    document_id: str
    document_title: Optional[str]
    section_id: Optional[str]
    section_title: Optional[str]
    label: Optional[str]
    specialty: Optional[str]
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    query_type: str = "general"

    @property
    def representative_text(self) -> str:
        return (self.chunks[0].get("text") or "")[:1000] if self.chunks else ""


REF_PATTERNS = [
    re.compile(r"\bdoi\s*:", re.IGNORECASE),
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\bet\s+al\.?", re.IGNORECASE),
    re.compile(r"\bvol\.\s*\d+", re.IGNORECASE),
    re.compile(r"\bpp\.\s*\d+", re.IGNORECASE),
]


def looks_like_references(text: str) -> bool:
    if not text:
        return False
    matches = sum(1 for pat in REF_PATTERNS if pat.search(text))
    if matches >= 2:
        return True
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    if len(years) >= 8:
        return True
    numbered = re.findall(r"(?m)^\s*\d{1,3}\.\s+[A-ZА-ЯЁ]", text)
    if len(numbered) >= 4:
        return True
    return False


def looks_like_toc_or_service(section_title: Optional[str], label: Optional[str]) -> bool:
    st = (section_title or "").lower()
    lb = (label or "").lower()
    if lb in JUNK_LABELS:
        return True
    return any(kw in st for kw in JUNK_SECTION_KEYWORDS)


def is_useful_chunk(
    chunk: Dict[str, Any],
    *,
    min_text_length: int,
    max_text_length: int,
) -> Tuple[bool, Optional[str]]:
    text = chunk.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        return False, "empty_text"
    n = len(text)
    if n < min_text_length:
        return False, "too_short"
    if n > max_text_length:
        return False, "too_long"
    if looks_like_toc_or_service(chunk.get("section_title"), chunk.get("label")):
        return False, "toc_or_service"
    if looks_like_references(text):
        return False, "references"
    return True, None

def _normalize_section_id(sid: Any) -> str:
    if sid is None or sid == "":
        return "__no_section__"
    return str(sid)


def group_into_candidates(chunks: Sequence[Dict[str, Any]]) -> List[Candidate]:
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        key = (
            str(c.get("document_id") or "__no_doc__"),
            _normalize_section_id(c.get("section_id")),
            str(c.get("label") or ""),
        )
        buckets[key].append(c)

    candidates: List[Candidate] = []
    for (doc_id, _sid, _lb), group in buckets.items():
        if not group:
            continue
        group.sort(key=lambda c: c.get("chunk_index") or 0)
        head = group[0]
        candidates.append(
            Candidate(
                document_id=doc_id,
                document_title=head.get("document_title"),
                section_id=head.get("section_id"),
                section_title=head.get("section_title"),
                label=head.get("label"),
                specialty=head.get("specialty"),
                chunks=group,
                query_type=label_to_query_type(head.get("label")),
            )
        )
    return candidates


def label_to_query_type(label: Optional[str]) -> str:
    if not label:
        return "general"
    key = str(label).strip().lower()
    return LABEL_TO_QUERY_TYPE.get(key, "general")


def score_candidate(c: Candidate) -> float:
    score = 0.0
    st = clean_topic(c.section_title)
    if st and len(st) >= 4:
        score += 2.0
    if len(st.split()) >= 2:
        score += 1.0
    if c.label:
        score += 1.0
    if c.query_type != "general":
        score += 1.0
    score += min(len(c.chunks) - 1, 3) * 0.5
    text_len = len(c.chunks[0].get("text") or "")
    if 200 <= text_len <= 2000:
        score += 1.0
    elif text_len > 4000:
        score -= 0.5
    return score


_NUMBERING_RE = re.compile(r"^\s*[\d\.\)]+\s+")


def clean_topic(title: Optional[str]) -> str:
    if not title:
        return ""
    t = _NUMBERING_RE.sub("", str(title)).strip()
    return t

_TOPIC_PREFIXES = [
    "клинические проявления",
    "лабораторная диагностика",
    "инструментальная диагностика",
    "дифференциальная диагностика",
    "диспансерное наблюдение",
    "диспансерный учет",
    "противопоказания к",
    "противопоказания для",
    "противопоказания при",
    "противопоказания",
    "показания к",
    "показания для",
    "показания при",
    "показания",
    "факторы риска",
    "методы лечения",
    "методы диагностики",
    "критерии диагностики",
    "критерии оценки",
    "диагностика",
    "лечение",
    "терапия",
    "симптомы",
    "признаки",
    "наблюдение",
    "обследование",
    "профилактика",
    "этиология",
    "патогенез",
    "осложнения",
    "реабилитация",
]


def extract_main_subject(section_title: Optional[str]) -> str:
    title = clean_topic(section_title).strip()
    if not title:
        return ""
    lower = title.lower()
    for pref in _TOPIC_PREFIXES:
        if lower.startswith(pref):
            tail = title[len(pref):].lstrip(" :,.-—–").strip()
            if len(tail) >= 3:
                return tail
    return title


def extract_section_keywords(c: Candidate, max_kw: int = 8) -> List[str]:
    seen: List[str] = []
    seen_lower: set = set()

    def _add(word: str) -> None:
        w = word.strip(".,;:()«»\"' \t").lower()
        if not w or len(w) < 4 or w in RU_STOPWORDS:
            return
        if w in seen_lower:
            return
        seen.append(word.strip(".,;:()«»\"' \t"))
        seen_lower.add(w)

    title = clean_topic(c.section_title)
    for w in re.findall(r"[А-Яа-яЁёA-Za-z\-]{4,}", title):
        _add(w)
        if len(seen) >= max_kw:
            return seen

    head_text = (c.chunks[0].get("text") or "")[:600]
    for w in re.findall(r"[А-ЯЁ][а-яё\-]{4,}", head_text):
        _add(w)
        if len(seen) >= max_kw:
            return seen

    return seen[:max_kw]

def allocate_targets(num_queries: int, distribution: Dict[str, float]) -> Dict[str, int]:
    raw = {k: num_queries * v for k, v in distribution.items()}
    floored = {k: int(v) for k, v in raw.items()}
    used = sum(floored.values())
    remainder = num_queries - used
    fractions = sorted(
        ((k, raw[k] - floored[k]) for k in raw),
        key=lambda x: x[1],
        reverse=True,
    )
    for k, _ in fractions[:remainder]:
        floored[k] += 1
    return floored


def stratified_select(
    candidates: List[Candidate],
    *,
    num_queries: int,
    max_queries_per_document: int,
    min_documents: int,
    rng: random.Random,
) -> List[Candidate]:
    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda c: c.quality_score, reverse=True)

    selected: List[Candidate] = []
    doc_counts: Counter = Counter()
    used_idx: set = set()

    if min_documents > 0:
        for i, c in enumerate(candidates_sorted):
            if len(selected) >= min(min_documents, num_queries):
                break
            if doc_counts[c.document_id] > 0:
                continue
            selected.append(c)
            used_idx.add(i)
            doc_counts[c.document_id] += 1

    qt_targets = allocate_targets(num_queries, QUERY_TYPE_DISTRIBUTION)
    qt_counts: Counter = Counter(c.query_type for c in selected)

    remaining_pool = [
        (i, c) for i, c in enumerate(candidates_sorted) if i not in used_idx
    ]
    rng.shuffle(remaining_pool)
    remaining_pool.sort(key=lambda ic: ic[1].quality_score, reverse=True)

    for qt, target in sorted(qt_targets.items(), key=lambda x: -x[1]):
        if qt_counts[qt] >= target:
            continue
        for i, c in list(remaining_pool):
            if len(selected) >= num_queries:
                return selected
            if c.query_type != qt:
                continue
            if doc_counts[c.document_id] >= max_queries_per_document:
                continue
            selected.append(c)
            used_idx.add(i)
            doc_counts[c.document_id] += 1
            qt_counts[qt] += 1
            if qt_counts[qt] >= target:
                break

    if len(selected) < num_queries:
        for i, c in enumerate(candidates_sorted):
            if i in used_idx:
                continue
            if doc_counts[c.document_id] >= max_queries_per_document:
                continue
            selected.append(c)
            used_idx.add(i)
            doc_counts[c.document_id] += 1
            if len(selected) >= num_queries:
                break

    return selected[:num_queries]


def assign_difficulties(
    selected: List[Candidate], rng: random.Random
) -> List[str]:
    n = len(selected)
    targets = allocate_targets(n, DIFFICULTY_DISTRIBUTION)
    pool: List[str] = []
    for diff, cnt in targets.items():
        pool.extend([diff] * cnt)
    rng.shuffle(pool)
    if len(pool) < n:
        pool.extend(["medium"] * (n - len(pool)))
    return pool[:n]


def build_query_text(
    c: Candidate, difficulty: str, rng: random.Random
) -> str:
    qt = c.query_type if (c.query_type, difficulty) in TEMPLATES else "general"
    templates = TEMPLATES.get((qt, difficulty)) or TEMPLATES.get((qt, "medium")) or [
        "клинические рекомендации {topic}"
    ]
    template = rng.choice(templates)

    topic = extract_main_subject(c.section_title)
    if not topic or len(topic) < 4:
        topic = (c.document_title or "клинических рекомендациях").strip()
        topic = re.sub(r"^клинические рекомендации:\s*", "", topic, flags=re.IGNORECASE).strip()

    return re.sub(r"\s+", " ", template.format(topic=topic.lower())).strip()


def build_source_evidence(c: Candidate, max_items: int = 3) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ch in c.chunks[:max_items]:
        text = (ch.get("text") or "").strip()
        items.append(
            {
                "chunk_id": ch.get("id"),
                "document_id": ch.get("document_id"),
                "document_title": ch.get("document_title"),
                "section_title": ch.get("section_title"),
                "page_start": ch.get("page_start"),
                "page_end": ch.get("page_end"),
                "evidence_text_preview": text[:300],
            }
        )
    return items


def build_query_record(
    c: Candidate,
    *,
    query_id: str,
    difficulty: str,
    query_text: str,
) -> Dict[str, Any]:
    expected_chunk_ids = [ch.get("id") for ch in c.chunks if ch.get("id")]
    expected_section_ids = (
        [c.section_id] if c.section_id not in (None, "") else []
    )
    expected_section_titles = (
        [c.section_title] if c.section_title else []
    )
    expected_labels = [c.label] if c.label else []
    expected_specialties = [c.specialty] if c.specialty else []

    relevance_level = "chunk" if len(c.chunks) == 1 else "section"
    record: Dict[str, Any] = {
        "query_id": query_id,
        "query": query_text,
        "query_type": c.query_type,
        "difficulty": difficulty,
        "expected_document_ids": [c.document_id] if c.document_id else [],
        "expected_chunk_ids": expected_chunk_ids,
        "expected_section_ids": expected_section_ids,
        "expected_section_titles": expected_section_titles,
        "expected_section_keywords": extract_section_keywords(c),
        "expected_labels": expected_labels,
        "expected_specialties": expected_specialties,
        "relevance_level": relevance_level,
        "source_evidence": build_source_evidence(c),
        "comment": (
            f"auto: запрос про '{clean_topic(c.section_title) or c.document_title}'"
            f" (label={c.label}, qtype={c.query_type})"
        ),
        "review_status": "auto_generated",
        "requiring_human_review": True,
    }
    return record


def write_candidates_csv(candidates: List[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "quality_score",
                "document_id",
                "document_title",
                "section_id",
                "section_title",
                "label",
                "query_type",
                "specialty",
                "num_chunks",
                "first_chunk_id",
                "first_chunk_text_preview",
            ]
        )
        for c in candidates:
            head = c.chunks[0] if c.chunks else {}
            writer.writerow(
                [
                    f"{c.quality_score:.3f}",
                    c.document_id,
                    c.document_title or "",
                    c.section_id or "",
                    c.section_title or "",
                    c.label or "",
                    c.query_type,
                    c.specialty or "",
                    len(c.chunks),
                    head.get("id", ""),
                    (head.get("text") or "")[:200].replace("\n", " "),
                ]
            )


def generate_eval_queries(
    *,
    chunks_path: str | Path,
    output_path: str | Path,
    candidates_path: str | Path,
    num_queries: int,
    max_queries_per_document: int,
    min_documents: int,
    min_text_length: int,
    max_text_length: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    chunks = load_chunks_any(chunks_path)
    logger.info("Loaded %d raw chunks from %s", len(chunks), chunks_path)

    kept: List[Dict[str, Any]] = []
    drop_reasons: Counter = Counter()
    for ch in chunks:
        ok, reason = is_useful_chunk(
            ch,
            min_text_length=min_text_length,
            max_text_length=max_text_length,
        )
        if ok:
            kept.append(ch)
        else:
            drop_reasons[reason or "unknown"] += 1
    logger.info(
        "After filtering: kept=%d dropped=%d reasons=%s",
        len(kept),
        len(chunks) - len(kept),
        dict(drop_reasons),
    )

    candidates = group_into_candidates(kept)
    for c in candidates:
        c.quality_score = score_candidate(c)
    logger.info(
        "Built %d candidate groups from %d chunks (avg group size = %.1f)",
        len(candidates),
        len(kept),
        len(kept) / max(len(candidates), 1),
    )

    candidates_sorted = sorted(candidates, key=lambda c: c.quality_score, reverse=True)
    write_candidates_csv(candidates_sorted, Path(candidates_path))
    logger.info("Saved candidates -> %s", candidates_path)

    selected = stratified_select(
        candidates_sorted,
        num_queries=num_queries,
        max_queries_per_document=max_queries_per_document,
        min_documents=min_documents,
        rng=rng,
    )
    logger.info("Selected %d candidates (target=%d)", len(selected), num_queries)
    if len(selected) < num_queries:
        logger.warning(
            "Selected fewer queries than requested (%d < %d). "
            "Causes: not enough unique documents/labels or per-doc cap is too tight.",
            len(selected),
            num_queries,
        )

    rng.shuffle(selected)
    difficulties = assign_difficulties(selected, rng)

    records: List[Dict[str, Any]] = []
    for idx, (c, diff) in enumerate(zip(selected, difficulties), start=1):
        qid = f"q{idx:03d}"
        query_text = build_query_text(c, diff, rng)
        rec = build_query_record(c, query_id=qid, difficulty=diff, query_text=query_text)
        records.append(rec)

    write_jsonl(output_path, records)
    logger.info("Saved %d eval queries -> %s", len(records), output_path)

    stats = compute_stats(records)
    logger.info("Stats by query_type: %s", stats["by_query_type"])
    logger.info("Stats by difficulty: %s", stats["by_difficulty"])
    logger.info("Stats by relevance_level: %s", stats["by_relevance_level"])
    logger.info("Unique expected documents: %d", stats["unique_documents"])

    return {
        "num_queries": len(records),
        "candidates": len(candidates),
        "stats": stats,
        "output_path": str(output_path),
        "candidates_path": str(candidates_path),
    }


def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_qt: Counter = Counter(r["query_type"] for r in records)
    by_diff: Counter = Counter(r["difficulty"] for r in records)
    by_rel: Counter = Counter(r["relevance_level"] for r in records)
    docs: set = set()
    for r in records:
        for d in r.get("expected_document_ids") or []:
            docs.add(d)
    return {
        "by_query_type": dict(by_qt.most_common()),
        "by_difficulty": dict(by_diff.most_common()),
        "by_relevance_level": dict(by_rel.most_common()),
        "unique_documents": len(docs),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-generate retrieval evaluation queries (drafts for human review)"
    )
    parser.add_argument("--chunks-path", required=True, type=str)
    parser.add_argument(
        "--output-path",
        default="data/retrieval_eval_queries_v1.jsonl",
        type=str,
    )
    parser.add_argument(
        "--candidates-path",
        default="outputs/reports/retrieval_eval_candidates.csv",
        type=str,
    )
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--max-queries-per-document", type=int, default=3)
    parser.add_argument("--min-documents", type=int, default=15)
    parser.add_argument("--min-text-length", type=int, default=180)
    parser.add_argument("--max-text-length", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_path).parent
    if str(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        verbose=args.verbose,
        log_file=Path(args.candidates_path).with_suffix(".log"),
        name="generate_eval",
    )
    try:
        info = generate_eval_queries(
            chunks_path=args.chunks_path,
            output_path=args.output_path,
            candidates_path=args.candidates_path,
            num_queries=args.num_queries,
            max_queries_per_document=args.max_queries_per_document,
            min_documents=args.min_documents,
            min_text_length=args.min_text_length,
            max_text_length=args.max_text_length,
            seed=args.seed,
        )
        logger.info("Done. info=%s", info)
    except Exception:
        logger.exception("Generation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
