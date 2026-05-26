import fitz
import re
import json
import uuid
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterable
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity






SPECIALTY = "pediatrics"




DEFAULT_EMBEDDING_MODEL: Optional[str] = None




DOCUMENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "КР295_4.pdf": {
        "document_id": "kr295_4",
        "document_title": "Клинические рекомендации: Мигрень",
        "topic": "мигрень",
        "specialty": "neurology",
        "term_expansions": {
            "М": "мигрень",
            "ГБ": "головная боль",
            "МбА": "мигрень без ауры",
            "МА": "мигрень с аурой",
            "ХМ": "хроническая мигрень",
            "ГБН": "головная боль напряжения",
            "ЛИГБ": "лекарственно-индуцированная головная боль",
            "ТИА": "транзиторная ишемическая атака",
            "ТМО": "твердая мозговая оболочка",
            "ТВС": "тригемино-васкулярная система",
        },
    },

    "КР607.pdf": {
        "document_id": "kr607",
        "document_title": "Клинические рекомендации: ...",  
        "topic": None,
        "specialty": SPECIALTY,
        "term_expansions": {},
    },
}

VALID_SECTION_KEYWORDS = [
    "этиология",
    "патогенез",
    "эпидемиология",
    "классификация",
    "диагностика",
    "лечение",
    "профилактика",
    "реабилитация",
    "клиническая картина",
    "инструментальная диагностика",
    "лабораторная диагностика",
    "диспансерное наблюдение",
]

EXCLUDED_VALID_SECTION_KEYWORDS = [
    "организация оказания",
    "дополнительная информация",
]

STOP_SECTION_KEYWORDS = [
    "список литературы",
    "литература",
    "библиограф",
    "приложение",
    "критерии оценки качества",
    "состав рабочей группы",
    "рабочая группа",
    "методология разработки",
    "целевая аудитория",
    "порядок обновления",
    "конфликт интересов",
    "персональный состав",
    "авторы",
    "разработчики",
]



TERM_EXPANSIONS = {
    "ЖП": "желчный пузырь",
    "ОАХ": "острый акалькулезный холецистит",
    "ХрХ": "хронический холецистит",
    "ОРВИ": "острая респираторная вирусная инфекция",
    "ОРИ": "острая респираторная инфекция",
    "ОХ": "острый холецистит",
    "ЛХЭ": "лапароскопическая холецистэктомия",
    "ХЭ": "холецистэктомия",
    "ЧЧХЦС": "чрескожная чреспеченочная холецистостомия",
}

MIN_SENTENCE_LEN = 40
MIN_CHUNK_TEXT_LEN = 150
KEEP_EVIDENCE_LEVELS = True

ALLOWED_LABELS = {
    "epidemiology",
    "classification",
    "symptoms",
    "diagnosis",
    "treatment",
    "prevention",
    "rehabilitation",
    "etiology",
    "pathogenesis",
    "complications",
    "unknown",
}

LABEL_ALIASES = {
    "general": "unknown",
    "definition": "unknown",
    "etiology_pathogenesis": "etiology",
}






@dataclass
class Line:
    text: str
    page: int
    y: float


@dataclass
class Section:
    id: str
    title: str
    page_start: int


@dataclass
class SentenceUnit:
    text: str
    page_start: int
    page_end: int
    section_id: str
    section_title: str
    label: str






def normalize_spaces(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_document_id(pdf_path: str, document_title: Optional[str] = None) -> str:
    """
    Стабильный короткий id документа.

    Пример:
    КР540_3.pdf -> kr540_3
    """
    base = Path(pdf_path).stem or normalize_spaces(document_title) or "document"
    base = base.lower()
    base = base.replace("кр", "kr")
    base = re.sub(r"[^a-zа-яё0-9]+", "_", base, flags=re.IGNORECASE)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "document"


def make_content_hash(embedding_text: str) -> str:
    """
    Хэш от embedding_text, потому что именно он потом уходит в embedding.
    Если embedding_text не изменился — embedding можно не пересчитывать.
    """
    normalized = normalize_spaces(embedding_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_noise_line(text: str) -> bool:
    t = text.lower().strip()

    if not t:
        return True

    if re.fullmatch(r"\d{1,4}", t):
        return True

    if re.search(r"\.{5,}", t):
        return True

    noise_phrases = [
        "оглавление",
        "утверждено",
        "список литературы",
    ]

    if any(x in t for x in noise_phrases):
        return True

    return False


def strip_leading_header_garbage(text: str) -> str:
    text = normalize_spaces(text)

    patterns = [
        r"^или\s+состояний\)\s+",
        r"^состояний\)\s+",
        r"^группы\s+заболеваний\s+и\s+состояний\)\s+",
        r"^медицинские\s+показания\s+и\s+противопоказания\s+к\s+применению\s+методов\s+[а-яА-ЯёЁ\s,\-]{0,140}",
        r"^показания\s+и\s+противопоказания\s+к\s+применению\s+методов\s+[а-яА-ЯёЁ\s,\-]{0,140}",
        r"^противопоказания\s+к\s+применению\s+методов\s+[а-яА-ЯёЁ\s,\-]{0,140}",
        r"^в\s+том\s+числе\s+основанных\s+на\s+использовании\s+природных\s+лечебных\s+факторов\s+",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return text


def clean_text(text: str) -> str:
    text = normalize_spaces(text)
    text = strip_leading_header_garbage(text)

    text = re.sub(r"[•▪◦●]", " ", text)
    text = text.replace("**", "")
    text = text.replace("#", "")
    text = text.replace("- ", "")

    text = re.sub(r"(\d{2})(\d{2})%", r"\1-\2%", text)
    text = re.sub(r"([а-яА-Я])([0-9])", r"\1 \2", text)
    text = re.sub(r"(\d+\.\d+)([А-ЯA-Z])", r"\1 \2", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\b\d+\.\b", "", text)

    
    replacements = [
        (r"\bэкстраи\s+", "экстра- и "),
        (r"\bинтраи\s+", "интра- и "),
        (r"\bпричинноследственная\b", "причинно-следственная"),
        (r"\bсердечнососудист", "сердечно-сосудист"),
        (r"\bоперационноанестезиолог", "операционно-анестезиолог"),
        (r"\bинструментальнолаборатор", "инструментально-лаборатор"),
        (r"\bпеченочнодвенадцатиперстн", "печеночно-двенадцатиперстн"),
        (r"\bреспираторносинцитиальн", "респираторно-синцитиальн"),
        (r"\bДНКсодержащ", "ДНК-содержащ"),
        (r"\bРНКсодержащ", "РНК-содержащ"),
        (r"\bврачомхирургом\b", "врачом-хирургом"),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(
        r"Адренергические показаны при сохраняющейся гипотонии, несмотря на и дофаминергические средства \(C01CA\) проводимую адекватную инфузионную терапию",
        "Адренергические и дофаминергические средства (C01CA) показаны при сохраняющейся гипотонии, несмотря на проводимую адекватную инфузионную терапию",
        text,
    )

    text = re.sub(r"\bпервые\s+57\s+сут", "первые 5-7 сут", text)
    text = re.sub(r"\b39\s+39,5°С", "39-39,5°С", text)
    text = re.sub(r"\b109/л\s+109/л\b", "10^9/л", text)
    text = re.sub(r"\b109/л\b", "10^9/л", text)
    text = re.sub(r"\b15\s+х\s+в сочетании\b", "15 × 10^9/л в сочетании", text)
    text = re.sub(r"\b10\s+х\s+и более\b", "10 × 10^9/л и более", text)
    text = re.sub(r"\bo\s+([а-яА-Я])", r"\1", text)

    if not KEEP_EVIDENCE_LEVELS:
        text = re.sub(
            r"\(?Уровень убедительности рекомендаций\s*[AАBВCС]\s*;\s*уровень достоверности доказательств\s*\d+\)?\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)

    return normalize_spaces(text)






def extract_section_title(line: str) -> Optional[Dict[str, str]]:
    line = normalize_spaces(line)

    match = re.match(r"^(\d+(?:\.\d+)*)(?:\.)?\s+(.+)$", line)
    if not match:
        return None

    section_id = match.group(1)
    title = match.group(2).strip()

    if len(title) > 220:
        return None

    if not title:
        return None

    if title[0].islower():
        return None

    title = re.split(r"\s*\(|\s+—\s+|\s+-\s+", title)[0]
    title = " ".join(title.split()[:18])

    return {
        "id": section_id,
        "title": title.strip(),
    }


def is_valid_section(title: Optional[str]) -> bool:
    if not title:
        return False

    t = title.lower()

    if any(x in t for x in EXCLUDED_VALID_SECTION_KEYWORDS):
        return False

    return any(k in t for k in VALID_SECTION_KEYWORDS)


def is_stop_section(title: Optional[str]) -> bool:
    if not title:
        return False

    t = title.lower()
    return any(k in t for k in STOP_SECTION_KEYWORDS)


def looks_like_heading(text: str) -> bool:
    t = normalize_spaces(text)

    if not t or len(t) > 160:
        return False

    lower = t.lower()

    heading_words = [
        "этиология",
        "патогенез",
        "эпидемиология",
        "классификация",
        "диагностика",
        "инструментальная диагностика",
        "лабораторная диагностика",
        "лечение",
        "профилактика",
        "реабилитация",
        "клиническая картина",
        "медицинская реабилитация",
        "диспансерное наблюдение",
        "антибактериальная терапия",
        "хирургическое лечение",
        "консервативное лечение",
        "критерии установления диагноза",
    ]

    if any(w in lower for w in heading_words):
        if not re.search(r"[.!?]$", t):
            return True

    if re.match(r"^(Пациентам|Рекомендуется|Не рекомендуется|Следует|Комментарии)\b", t):
        return False

    if len(t) < 80 and t[0].isupper() and not re.search(r"[.!?]$", t):
        return True

    return False






def extract_lines_from_page(page, page_number: int) -> List[Line]:
    words = page.get_text("words")
    lines_by_y: Dict[float, List[Any]] = {}

    for word in words:
        x0, y0, x1, y1, text, *_ = word
        y_key = round(y0, 1)
        lines_by_y.setdefault(y_key, []).append((x0, text))

    lines: List[Line] = []

    for y in sorted(lines_by_y.keys()):
        words_sorted = sorted(lines_by_y[y], key=lambda x: x[0])
        text = " ".join(w[1] for w in words_sorted)
        text = normalize_spaces(text)

        if is_noise_line(text):
            continue

        lines.append(Line(text=text, page=page_number, y=y))

    return lines


def extract_pdf_lines(pdf_path: str) -> List[Line]:
    doc = fitz.open(pdf_path)
    all_lines: List[Line] = []

    for page_number, page in enumerate(doc, start=1):
        page_lines = extract_lines_from_page(page, page_number)
        all_lines.extend(page_lines)

    doc.close()
    return all_lines


def looks_like_broken_text_layer(lines: List[Line]) -> bool:
    """
    Диагностика PDF с битым текстовым слоем.
    Например, визуально русский текст есть, а fitz извлекает '%>AB>O=85...'.
    Это НЕ запускает OCR, только предупреждает.
    """
    sample_text = " ".join(line.text for line in lines[:300])
    sample_text = normalize_spaces(sample_text)

    if len(sample_text) < 300:
        return False

    letters = [ch for ch in sample_text if ch.isalpha()]
    if not letters:
        return True

    cyrillic_letters = [
        ch for ch in letters
        if "а" <= ch.lower() <= "я" or ch.lower() == "ё"
    ]

    cyrillic_ratio = len(cyrillic_letters) / max(len(letters), 1)

    
    return cyrillic_ratio < 0.2






def should_merge_lines_smart(prev: str, current: str) -> bool:
    prev = prev.strip()
    current = current.strip()

    if not prev or not current:
        return False

    if extract_section_title(prev) or extract_section_title(current):
        return False

    if looks_like_heading(current):
        return False

    if re.match(r"^[\-–•▪◦●]\s+", current):
        return not re.search(r"[.!?]$", prev)

    if re.match(r"^(Пациентам|Рекомендуется|Не рекомендуется|Следует)\b", current):
        if re.search(r"[.!?]$", prev):
            return False

    if re.match(
        r"^(и|или|а|но|при|с|без|для|у|в|на|по|от|до|из|к|что|который|которая|которые)\b",
        current.lower(),
    ):
        return True

    if re.search(r"[,;:]$", prev):
        return True

    if re.search(
        r"\b(и|или|а|но|при|с|без|для|у|в|на|по|от|до|из|к|как|что|который|которая|которые|его|ее|их)$",
        prev.lower(),
    ):
        return True

    if not re.search(r"[.!?]$", prev):
        return True

    if current[0].islower():
        return True

    return False


def build_structured_paragraphs(lines: List[Line]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    buffer_text = ""
    buffer_page_start = None
    buffer_page_end = None

    def flush_buffer() -> None:
        nonlocal buffer_text, buffer_page_start, buffer_page_end

        if buffer_text:
            items.append({
                "type": "paragraph",
                "text": normalize_spaces(buffer_text),
                "page_start": buffer_page_start,
                "page_end": buffer_page_end,
            })

        buffer_text = ""
        buffer_page_start = None
        buffer_page_end = None

    for line in lines:
        text = line.text.strip()

        section = extract_section_title(text)

        if section:
            flush_buffer()
            items.append({
                "type": "section",
                "id": section["id"],
                "title": section["title"],
                "page_start": line.page,
                "page_end": line.page,
            })
            continue

        if not buffer_text:
            buffer_text = text
            buffer_page_start = line.page
            buffer_page_end = line.page
            continue

        if should_merge_lines_smart(buffer_text, text):
            buffer_text += " " + text
            buffer_page_end = line.page
        else:
            flush_buffer()
            buffer_text = text
            buffer_page_start = line.page
            buffer_page_end = line.page

    flush_buffer()
    return items






def split_into_sentences(text: str) -> List[str]:
    text = normalize_spaces(text)

    protected = {
        "т.е.": "т§е§",
        "т.к.": "т§к§",
        "т.ч.": "т§ч§",
        "в т.ч.": "в т§ч§",
        "и т.д.": "и т§д§",
        "и т.п.": "и т§п§",
        "др.": "др§",
        "рис.": "рис§",
        "табл.": "табл§",
        "стр.": "стр§",
        "им.": "им§",
        "г.": "г§",
        "мм рт.ст.": "мм рт§ст§",
    }

    for src, dst in protected.items():
        text = text.replace(src, dst)

    parts = re.split(r"(?<=[.!?])\s+", text)
    restored: List[str] = []

    for part in parts:
        for src, dst in protected.items():
            part = part.replace(dst, src)

        part = part.strip()
        if part:
            restored.append(part)

    return restored


def looks_truncated_sentence(sent: str) -> bool:
    s = sent.strip()
    low = s.lower()

    if not s:
        return True

    if len(s) < 80:
        return False

    exact_bad_endings = [
        r"\bот его$",
        r"\bвовремя$",
        r"\bбактериальная$",
        r"\bантибактериальная$",
        r"\bдо заражения ори$",
        r"\bс профилактической целью до заражения ори$",
        r"\bпри применении занамивира возможен бронхоспазм и другие аллергические реакции$",
    ]

    if any(re.search(p, low) for p in exact_bad_endings):
        return True

    if re.search(
        r"\b(от|до|из|для|при|с|без|в|на|по|к|и|или|а|но|его|ее|их|данным|целью|путем|посредством)$",
        low,
    ):
        return True

    return False


def is_bad_sentence(sent: str) -> bool:
    t = sent.lower().strip()

    if len(t) < MIN_SENTENCE_LEN:
        return True

    if re.search(r"(комментарий|рис\.|табл\.)", t):
        if "комментарии:" not in t:
            return True

    if re.search(r"\.{5,}", t):
        return True

    if looks_truncated_sentence(sent):
        return True

    return False






def normalize_label(label: Optional[str]) -> str:
    if not label:
        return "unknown"

    label = LABEL_ALIASES.get(label, label)

    if label not in ALLOWED_LABELS:
        return "unknown"

    return label


def classify(text: str) -> str:
    t = text.lower()

    scores = {
        "definition": sum(k in t for k in [
            "определение",
            "называется",
            "представляет собой",
        ]),
        "classification": sum(k in t for k in [
            "классификац",
            "класс",
            "степен",
            "стад",
            "форма",
            "вариант",
        ]),
        "symptoms": sum(k in t for k in [
            "боль",
            "симптом",
            "температур",
            "клиническ",
            "проявлен",
            "жалоб",
            "кашель",
            "насморк",
            "лихорад",
        ]),
        "diagnosis": sum(k in t for k in [
            "диагност",
            "анализ",
            "рентген",
            "узи",
            "исследован",
            "визуализац",
            "критери",
            "лейкоцит",
            "отоскоп",
        ]),
        "treatment": sum(k in t for k in [
            "лечение",
            "терап",
            "назнач",
            "препарат",
            "холецистэктом",
            "антибиотик",
            "инфузион",
            "коррекц",
            "осельтамивир",
            "парацетамол",
            "ибупрофен",
        ]),
        "etiology": sum(k in t for k in [
            "возбудител",
            "причин",
            "фактор",
            "этиолог",
            "предрасполага",
            "вирус",
            "бактери",
        ]),
        "pathogenesis": sum(k in t for k in [
            "патогенез",
            "механизм",
            "развивается вследствие",
            "приводящих к",
            "цитокин",
            "интерлейкин",
        ]),
        "complications": sum(k in t for k in [
            "осложн",
            "абсцесс",
            "свищ",
            "гангрен",
            "летальность",
            "пневмони",
            "отит",
        ]),
        "prevention": sum(k in t for k in [
            "профилактик",
            "предупрежден",
            "диспансер",
            "вакцинац",
            "иммунизац",
        ]),
        "rehabilitation": sum(k in t for k in [
            "реабилитац",
            "восстанов",
            "санатор",
            "физиотерап",
            "лфк",
        ]),
        "epidemiology": sum(k in t for k in [
            "частота",
            "распростран",
            "эпидемиолог",
            "случаев",
            "пациентов",
            "заболеваемость",
            "летальность",
        ]),
    }

    best = max(scores, key=scores.get)
    return normalize_label(best if scores[best] > 0 else "unknown")


def classify_with_section(text: str, section_title: str) -> str:
    st = section_title.lower()
    t = text.lower()

    if "этиология" in st and "патогенез" in st:
        if any(x in t for x in [
            "причин",
            "фактор",
            "возбудител",
            "предрасполага",
            "инфекц",
            "паразит",
            "аллерг",
            "вирус",
            "бактери",
        ]):
            return "etiology"

        if any(x in t for x in [
            "развивается",
            "воспал",
            "механизм",
            "привод",
            "формируется",
            "патогенез",
            "цитокин",
            "интерлейкин",
            "иммунитет",
        ]):
            return "pathogenesis"

        return "etiology"

    if "этиология" in st:
        return "etiology"

    if "патогенез" in st:
        return "pathogenesis"

    if "эпидемиология" in st:
        return "epidemiology"

    if "классификац" in st:
        return "classification"

    if "клиническая картина" in st:
        return "symptoms"

    if "диагностика" in st:
        return "diagnosis"

    if "лечение" in st:
        return "treatment"

    if "профилактика" in st or "диспансерное наблюдение" in st:
        return "prevention"

    if "реабилитац" in st or "санатор" in st:
        return "rehabilitation"

    return classify(text)


def extract_stage(text: str) -> Optional[str]:
    t = text.lower()

    acute_patterns = [
        r"\bоах\b",
        r"остр\w+\s+акалькулезн\w+\s+холецистит",
        r"остр\w+\s+бескаменн\w+\s+холецистит",
        r"остр\w+\s+холецистит",
    ]

    chronic_patterns = [
        r"\bхрх\b",
        r"хроническ\w+\s+акалькулезн\w+\s+холецистит",
        r"хроническ\w+\s+бескаменн\w+\s+холецистит",
        r"хроническ\w+\s+холецистит",
    ]

    acute = any(re.search(p, t) for p in acute_patterns)
    chronic = any(re.search(p, t) for p in chronic_patterns)

    if acute and chronic:
        return "acute_and_chronic"

    if acute:
        return "acute"

    if chronic:
        return "chronic"

    return None


def get_document_config(
    pdf_path: str,
    manual_document_title: Optional[str] = None,
    topic: Optional[str] = None,
    specialty: Optional[str] = None,
    manual_document_id: Optional[str] = None,
) -> Dict[str, Any]:
    pdf_name = Path(pdf_path).name
    pdf_stem = Path(pdf_path).stem

    cfg = DOCUMENT_CONFIG.get(pdf_name) or DOCUMENT_CONFIG.get(pdf_stem) or {}

    document_title = (
        manual_document_title
        or cfg.get("document_title")
        or f"Клинические рекомендации: {pdf_stem}"
    )

    document_id = (
        manual_document_id
        or cfg.get("document_id")
        or make_document_id(pdf_path, document_title=document_title)
    )

    return {
        "document_id": document_id,
        "document_title": document_title,
        "topic": topic if topic is not None else cfg.get("topic"),
        "specialty": specialty or cfg.get("specialty") or SPECIALTY,
        "term_expansions": cfg.get("term_expansions") or {},
    }


def get_term_expansions_for_text(
    text: str,
    extra_expansions: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    expansions = dict(TERM_EXPANSIONS)

    if extra_expansions:
        expansions.update(extra_expansions)

    result: Dict[str, str] = {}

    for term, expansion in expansions.items():
        if re.search(rf"\b{re.escape(term)}\b", text):
            result[term] = expansion

    return result


def build_embedding_text(
    document_title: str,
    section_id: str,
    section_title: str,
    label: str,
    text: str,
    topic: Optional[str] = None,
    term_expansions: Optional[Dict[str, str]] = None,
) -> str:
    parts = [
        f"Документ: {document_title}.",
    ]

    if topic:
        parts.append(f"Тема: {topic}.")

    section_repr = f"{section_id} {section_title}".strip()
    parts.append(f"Раздел: {section_repr}.")
    parts.append(f"Категория: {label}.")

    if term_expansions:
        abbreviations = "; ".join(
            f"{term} — {expansion}"
            for term, expansion in term_expansions.items()
        )
        parts.append(f"Сокращения: {abbreviations}.")

    parts.append(f"Текст: {text}")

    return " ".join(parts)


def extract_abbreviations_from_lines(
    lines: List[Line],
    max_pages: int = 6,
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    in_abbreviation_block = False

    header_markers = [
        "список сокращ",
        "перечень сокращ",
        "сокращения",
        "условные обозначения",
    ]

    for line in lines:
        if line.page > max_pages:
            break

        text = normalize_spaces(line.text)
        lower = text.lower()

        if any(marker in lower for marker in header_markers):
            in_abbreviation_block = True
            continue

        if not in_abbreviation_block:
            continue

        if extract_section_title(text):
            break

        if not text or len(text) > 240:
            continue

        parts = [p.strip() for p in re.split(r";\s*", text) if p.strip()]

        for part in parts:
            match = re.match(
                r"^([А-ЯЁA-Z][А-ЯЁA-Zа-яёA-Za-z0-9/-]{1,15})\s*[—–-]\s*(.{3,180})$",
                part,
            )

            if not match:
                match = re.match(
                    r"^([А-ЯЁA-Z][А-ЯЁA-Zа-яёA-Za-z0-9/-]{1,15})\s+(.{3,180})$",
                    part,
                )

            if not match:
                continue

            term = normalize_spaces(match.group(1)).strip(" .,:;()[]")
            expansion = normalize_spaces(match.group(2)).strip(" .,:;()[]")

            if not term or not expansion:
                continue

            if len(term) > 16 or len(expansion) < 3:
                continue

            if re.fullmatch(r"\d+", term):
                continue

            if len(re.sub(r"[^а-яА-ЯёЁa-zA-Z]", "", expansion)) < 3:
                continue

            if any(bad in expansion.lower() for bad in ["страница", "оглавление", "утвержден"]):
                continue

            result[term] = expansion

    return result






def extract_relevant_sentences(items: List[Dict[str, Any]]) -> List[SentenceUnit]:
    current_section: Optional[Section] = None
    relevant_sentences: List[SentenceUnit] = []

    for item in items:
        if item["type"] == "section":
            title = item["title"]

            if is_stop_section(title):
                current_section = None
                continue

            current_section = Section(
                id=item["id"],
                title=title,
                page_start=item["page_start"],
            )
            continue

        if item["type"] != "paragraph":
            continue

        if current_section is None:
            continue

        if not is_valid_section(current_section.title):
            continue

        cleaned = clean_text(item["text"])

        for sent in split_into_sentences(cleaned):
            sent = strip_leading_header_garbage(sent)

            if is_bad_sentence(sent):
                continue

            relevant_sentences.append(
                SentenceUnit(
                    text=sent,
                    page_start=item["page_start"],
                    page_end=item["page_end"],
                    section_id=current_section.id,
                    section_title=current_section.title,
                    label=classify_with_section(sent, current_section.title),
                )
            )

    return relevant_sentences






def chunk_sentences_with_overlap(
    sentences: List[SentenceUnit],
    target_chars: int = 1000,
    max_chars: int = 1600,
    min_chars: int = 450,
    overlap_chars: int = 300,
    max_page_span: int = 2,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current: List[SentenceUnit] = []

    def units_len(units: List[SentenceUnit]) -> int:
        return sum(len(s.text) for s in units) + max(0, len(units) - 1)

    def current_len() -> int:
        return units_len(current)

    def page_span(units: List[SentenceUnit]) -> int:
        if not units:
            return 0

        page_start = min(s.page_start for s in units)
        page_end = max(s.page_end for s in units)
        return page_end - page_start + 1

    def make_chunk(units: List[SentenceUnit]) -> Dict[str, Any]:
        labels = [normalize_label(s.label) for s in units]
        label = Counter(labels).most_common(1)[0][0]

        return {
            "text": " ".join(s.text for s in units),
            "page_start": min(s.page_start for s in units),
            "page_end": max(s.page_end for s in units),
            "section_id": units[0].section_id,
            "section_title": units[0].section_title,
            "label": normalize_label(label),
        }

    def get_overlap_units(units: List[SentenceUnit]) -> List[SentenceUnit]:
        if overlap_chars <= 0:
            return []

        result = []
        total = 0

        for unit in reversed(units):
            length = len(unit.text)

            if result and total + length > overlap_chars:
                break

            if not result and length > overlap_chars * 1.8:
                break

            result.append(unit)
            total += length

        return list(reversed(result))

    def flush_without_overlap() -> None:
        nonlocal current
        if current and current_len() >= min_chars:
            chunks.append(make_chunk(current))
        current = []

    def flush_with_overlap() -> None:
        nonlocal current
        if current and current_len() >= min_chars:
            chunks.append(make_chunk(current))
        current = get_overlap_units(current)

    for sent in sentences:
        if not current:
            current.append(sent)
            continue

        same_section = sent.section_id == current[0].section_id
        projected = current + [sent]
        projected_len = units_len(projected)
        projected_page_span = page_span(projected)

        if not same_section:
            flush_without_overlap()
            current = [sent]
            continue

        should_split = False

        if projected_page_span > max_page_span:
            should_split = True
        elif current_len() >= target_chars and projected_len > max_chars:
            should_split = True

        if should_split:
            flush_with_overlap()

            while current and page_span(current + [sent]) > max_page_span:
                current = current[1:]

            if current and units_len(current + [sent]) > max_chars:
                current = []

        current.append(sent)

    flush_without_overlap()
    return chunks






def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    threshold: float = 0.90,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    if len(chunks) == 1:
        return chunks

    texts = [c["text"] for c in chunks]

    try:
        vec = TfidfVectorizer().fit_transform(texts)
        sim_matrix = cosine_similarity(vec)
    except ValueError:
        return chunks

    keep = []
    removed = set()

    for i in range(len(chunks)):
        if i in removed:
            continue

        keep.append(chunks[i])

        for j in range(i + 1, len(chunks)):
            if j in removed:
                continue

            if chunks[i].get("section_id") != chunks[j].get("section_id"):
                continue

            if sim_matrix[i][j] > threshold:
                removed.add(j)

    return keep


def normalize_for_exact_dedup(text: str) -> str:
    text = normalize_spaces(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate_exact_final_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()

    for chunk in chunks:
        key = (
            chunk.get("document_id"),
            chunk.get("section_id"),
            normalize_for_exact_dedup(chunk.get("text") or ""),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(chunk)

    return result






def is_metadata_or_appendix_text(text: str) -> bool:
    lower = normalize_spaces(text).lower()

    strong_phrases = [
        "состав рабочей группы",
        "методология разработки",
        "целевая аудитория",
        "член общероссийской",
        "общероссийской общественной организации",
        "список литературы",
        "библиографический список",
        "конфликт интересов",
        "порядок обновления клинических рекомендаций",
        "персональный состав",
        "шкалы оценки, вопросники и другие оценочные инструменты",
        "перечень сокращений",
        "список сокращений",
    ]

    if any(phrase in lower for phrase in strong_phrases):
        return True

    soft_phrases = [
        "д.м.н.",
        "к.м.н.",
        "профессор",
        "доцент",
        "кафедра",
        "университет",
        "федеральное государственное",
        "научный центр",
        "общественная организация",
        "ассоциация",
        "экспертный совет",
        "разработчик",
    ]

    soft_hits = sum(phrase in lower for phrase in soft_phrases)
    if soft_hits >= 2:
        return True

    abbreviation_pairs = re.findall(
        r"\b[А-ЯЁA-Z]{2,}(?:-[А-ЯЁA-Zа-яёA-Za-z0-9]+)?\s*[—–-]\s*",
        text,
    )
    if len(abbreviation_pairs) >= 5:
        return True

    return False


def is_low_value_chunk(chunk: Dict[str, Any]) -> bool:
    text = normalize_spaces(chunk.get("text") or "")
    lower = text.lower()

    if not text:
        return True

    if is_metadata_or_appendix_text(text):
        return True

    if len(text) < MIN_CHUNK_TEXT_LEN:
        return True

    if re.fullmatch(r"[\d\s.,;:()\[\]{}\-–—/]+", text):
        return True

    letters = re.sub(r"[^а-яА-ЯёЁa-zA-Z]", "", text)
    if len(letters) / max(len(text), 1) < 0.25:
        return True

    service_phrases = [
        "министерство здравоохранения",
        "клинические рекомендации",
        "год утверждения",
        "версия",
        "профессиональная ассоциация",
        "российское общество",
        "оглавление",
    ]

    service_hits = sum(phrase in lower for phrase in service_phrases)
    if service_hits >= 2 and len(text) < 400:
        return True

    return False


def safe_page(value: Any, fallback: int = 1) -> int:
    try:
        page = int(value)
        return page if page > 0 else fallback
    except Exception:
        return fallback


def make_final_chunk(
    raw_chunk: Dict[str, Any],
    pdf_path: str,
    doc_cfg: Dict[str, Any],
    chunk_index: int,
    embedding_model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    text = normalize_spaces(raw_chunk.get("text") or "")

    document_id = doc_cfg["document_id"]
    document_title = doc_cfg["document_title"]
    topic = doc_cfg.get("topic")
    specialty = doc_cfg.get("specialty") or SPECIALTY

    section_id = str(raw_chunk.get("section_id") or "").strip()
    section_title = normalize_spaces(
        raw_chunk.get("section_title") or "Неизвестный раздел"
    )

    label = normalize_label(raw_chunk.get("label"))

    page_start = safe_page(raw_chunk.get("page_start"), fallback=1)
    page_end = safe_page(raw_chunk.get("page_end"), fallback=page_start)

    if page_end < page_start:
        page_end = page_start

    term_expansions = get_term_expansions_for_text(
        text,
        extra_expansions=doc_cfg.get("term_expansions") or {},
    )

    embedding_text = build_embedding_text(
        document_title=document_title,
        section_id=section_id,
        section_title=section_title,
        label=label,
        text=text,
        topic=topic,
        term_expansions=term_expansions,
    )

    return {
        "id": str(uuid.uuid4()),
        "document_id": document_id,
        "chunk_index": chunk_index,
        "text": text,
        "embedding_text": embedding_text,
        "document_title": document_title,
        "section_id": section_id,
        "section_title": section_title,
        "label": label,
        "source": pdf_path,
        "page_start": page_start,
        "page_end": page_end,
        "specialty": specialty,
        "stage": extract_stage(text),
        "term_expansions": term_expansions,
        "content_hash": make_content_hash(embedding_text),
        "embedding_model": embedding_model,
    }






def debug_sections(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []

    for item in items:
        if item["type"] == "section":
            result.append({
                "section_id": item["id"],
                "section_title": item["title"],
                "page_start": item["page_start"],
                "is_valid": is_valid_section(item["title"]),
                "is_stop": is_stop_section(item["title"]),
            })

    return result


def save_json(data: Any, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_debug_sections(items: List[Dict[str, Any]], path: str = "sections_debug.json") -> None:
    save_json(debug_sections(items), path)


def is_suspicious_chunk(chunk: Dict[str, Any]) -> bool:
    text = chunk["text"]

    patterns = [
        r"\bпо данным Однако\b",
        r"\bна и\b",
        r"\bнесмотря на и\b",
        r"\bот его\b",
        r"\bпри помощи применения\b",
        r"\bпризнаком перфорации\b",
        r"\bврачомхирургом\b",
        r"\b57 суток\b",
        r"\bпеченочнодвенадцатиперстной\b",
        r"\b109/л\b",
        r"\bбактериальная$",
        r"\bвовремя Таким образом\b",
        r"^медицинские показания",
        r"^противопоказания к применению",
        r"^или состояний\)",
    ]

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    if len(text) > 2200 and chunk["page_end"] > chunk["page_start"]:
        return True

    return False


def save_debug_suspicious_chunks(
    chunks: List[Dict[str, Any]],
    path: str = "suspicious_chunks.json",
) -> None:
    suspicious = [c for c in chunks if is_suspicious_chunk(c)]
    save_json(suspicious, path)






def build_chunks_report(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks:
        return {
            "documents": 0,
            "document_ids": 0,
            "chunks_total": 0,
            "avg_chunk_len_chars": 0,
            "min_chunk_len_chars": 0,
            "max_chunk_len_chars": 0,
            "chunks_without_document_id": 0,
            "chunks_without_document_title": 0,
            "chunks_without_chunk_index": 0,
            "chunks_without_content_hash": 0,
            "chunks_without_section_title": 0,
            "chunks_with_unknown_section_title": 0,
            "chunks_without_page_start": 0,
            "chunks_with_label_unknown": 0,
        }

    lengths = [len(c.get("text") or "") for c in chunks]

    return {
        "documents": len(set(c.get("source") for c in chunks)),
        "document_ids": len(set(c.get("document_id") for c in chunks if c.get("document_id"))),
        "chunks_total": len(chunks),
        "avg_chunk_len_chars": round(sum(lengths) / len(lengths)),
        "min_chunk_len_chars": min(lengths),
        "max_chunk_len_chars": max(lengths),
        "chunks_without_document_id": sum(not c.get("document_id") for c in chunks),
        "chunks_without_document_title": sum(not c.get("document_title") for c in chunks),
        "chunks_without_chunk_index": sum(c.get("chunk_index") is None for c in chunks),
        "chunks_without_content_hash": sum(not c.get("content_hash") for c in chunks),
        "chunks_without_section_title": sum(not c.get("section_title") for c in chunks),
        "chunks_with_unknown_section_title": sum(c.get("section_title") == "Неизвестный раздел" for c in chunks),
        "chunks_without_page_start": sum(c.get("page_start") is None for c in chunks),
        "chunks_with_label_unknown": sum(c.get("label") == "unknown" for c in chunks),
    }


def save_report(report: Dict[str, Any], path: str = "chunks_report.json") -> None:
    save_json(report, path)


def save_random_review_sample(
    chunks: List[Dict[str, Any]],
    path: str = "chunks_review_sample.json",
    sample_size: int = 20,
    seed: int = 42,
) -> None:
    random.seed(seed)
    sample = random.sample(chunks, k=min(sample_size, len(chunks)))
    save_json(sample, path)


def save_chunks(chunks: List[Dict[str, Any]], path: str = "chunks.json") -> None:
    save_json(chunks, path)






def parse_pdf_to_chunks(
    pdf_path: str,
    target_chars: int = 1000,
    max_chars: int = 1600,
    min_chars: int = 450,
    overlap_chars: int = 300,
    max_page_span: int = 2,
    dedup_threshold: float = 0.90,
    save_sections_debug: bool = False,
    save_suspicious_debug: bool = False,
    manual_document_title: Optional[str] = None,
    topic: Optional[str] = None,
    specialty: Optional[str] = None,
    manual_document_id: Optional[str] = None,
    embedding_model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
) -> List[Dict[str, Any]]:
    doc_cfg = get_document_config(
        pdf_path=pdf_path,
        manual_document_title=manual_document_title,
        topic=topic,
        specialty=specialty,
        manual_document_id=manual_document_id,
    )

    lines = extract_pdf_lines(pdf_path)

    if looks_like_broken_text_layer(lines):
        print(
            f"⚠️  Похоже, у PDF сломан текстовый слой: {pdf_path}. "
            "Для этого файла нужен OCR перед парсингом."
        )

    extracted_abbreviations = extract_abbreviations_from_lines(lines)
    if extracted_abbreviations:
        doc_cfg["term_expansions"] = {
            **extracted_abbreviations,
            **(doc_cfg.get("term_expansions") or {}),
        }

    structured_items = build_structured_paragraphs(lines)

    if save_sections_debug:
        save_debug_sections(structured_items)

    relevant_sentences = extract_relevant_sentences(structured_items)

    chunks = chunk_sentences_with_overlap(
        relevant_sentences,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars,
        max_page_span=max_page_span,
    )

    chunks = deduplicate_chunks(chunks, threshold=dedup_threshold)

    final: List[Dict[str, Any]] = []

    for ch in chunks:
        if is_low_value_chunk(ch):
            continue

        final_chunk = make_final_chunk(
            raw_chunk=ch,
            pdf_path=pdf_path,
            doc_cfg=doc_cfg,
            chunk_index=len(final),
            embedding_model=embedding_model,
        )

        if is_low_value_chunk(final_chunk):
            continue

        final.append(final_chunk)

    final = deduplicate_exact_final_chunks(final)

    
    for idx, chunk in enumerate(final):
        chunk["chunk_index"] = idx

    if save_suspicious_debug:
        save_debug_suspicious_chunks(final)

    return final


def find_pdf_files(input_dir: str, recursive: bool = True) -> List[Path]:
    root = Path(input_dir)

    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []

    if not root.exists():
        raise FileNotFoundError(f"Путь не найден: {input_dir}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(root.glob(pattern))


def parse_input_to_chunks(
    input_path: str,
    target_chars: int = 1000,
    max_chars: int = 1600,
    min_chars: int = 450,
    overlap_chars: int = 300,
    max_page_span: int = 2,
    dedup_threshold: float = 0.90,
    recursive: bool = True,
    save_sections_debug: bool = False,
    save_suspicious_debug: bool = False,
    manual_document_title: Optional[str] = None,
    topic: Optional[str] = None,
    specialty: Optional[str] = None,
    manual_document_id: Optional[str] = None,
    embedding_model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
) -> List[Dict[str, Any]]:
    path = Path(input_path)

    if path.is_file():
        return parse_pdf_to_chunks(
            pdf_path=str(path),
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_chars=overlap_chars,
            max_page_span=max_page_span,
            dedup_threshold=dedup_threshold,
            save_sections_debug=save_sections_debug,
            save_suspicious_debug=save_suspicious_debug,
            manual_document_title=manual_document_title,
            topic=topic,
            specialty=specialty,
            manual_document_id=manual_document_id,
            embedding_model=embedding_model,
        )

    pdf_files = find_pdf_files(input_path, recursive=recursive)
    all_chunks: List[Dict[str, Any]] = []

    for pdf_file in pdf_files:
        chunks = parse_pdf_to_chunks(
            pdf_path=str(pdf_file),
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_chars=overlap_chars,
            max_page_span=max_page_span,
            dedup_threshold=dedup_threshold,
            save_sections_debug=False,
            save_suspicious_debug=False,
            manual_document_title=None,
            topic=None,
            specialty=None,
            manual_document_id=None,
            embedding_model=embedding_model,
        )
        all_chunks.extend(chunks)

    return all_chunks






def make_mirrored_output_paths(
    pdf_file: Path,
    input_root: Path,
    output_root: Path,
    chunks_suffix: str = ".chunks.json",
) -> Dict[str, Path]:
    try:
        relative_pdf = pdf_file.relative_to(input_root)
    except ValueError:
        relative_pdf = Path(pdf_file.name)

    output_dir = output_root / relative_pdf.parent
    stem = pdf_file.stem

    return {
        "chunks": output_dir / f"{stem}{chunks_suffix}",
        "report": output_dir / f"{stem}.report.json",
        "review_sample": output_dir / f"{stem}.review_sample.json",
        "suspicious": output_dir / f"{stem}.suspicious.json",
        "sections_debug": output_dir / f"{stem}.sections_debug.json",
    }


def parse_folder_to_separate_chunk_files(
    input_dir: str,
    output_dir: str,
    target_chars: int = 1000,
    max_chars: int = 1600,
    min_chars: int = 450,
    overlap_chars: int = 300,
    max_page_span: int = 2,
    dedup_threshold: float = 0.90,
    recursive: bool = True,
    save_sections_debug: bool = False,
    save_suspicious_debug: bool = False,
    review_sample_size: int = 20,
    chunks_suffix: str = ".chunks.json",
    continue_on_error: bool = True,
    embedding_model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    if not input_root.exists():
        raise FileNotFoundError(f"Папка не найдена: {input_dir}")

    if not input_root.is_dir():
        raise ValueError(f"Для отдельного сохранения нужна папка, получен файл: {input_dir}")

    pdf_files = find_pdf_files(str(input_root), recursive=recursive)

    results: List[Dict[str, Any]] = []
    chunks_total = 0
    files_ok = 0
    files_failed = 0

    for pdf_file in pdf_files:
        output_paths = make_mirrored_output_paths(
            pdf_file=pdf_file,
            input_root=input_root,
            output_root=output_root,
            chunks_suffix=chunks_suffix,
        )

        try:
            chunks = parse_pdf_to_chunks(
                pdf_path=str(pdf_file),
                target_chars=target_chars,
                max_chars=max_chars,
                min_chars=min_chars,
                overlap_chars=overlap_chars,
                max_page_span=max_page_span,
                dedup_threshold=dedup_threshold,
                save_sections_debug=False,
                save_suspicious_debug=False,
                manual_document_title=None,
                topic=None,
                specialty=None,
                manual_document_id=None,
                embedding_model=embedding_model,
            )

            save_chunks(chunks, str(output_paths["chunks"]))

            report = build_chunks_report(chunks)
            save_report(report, str(output_paths["report"]))

            save_random_review_sample(
                chunks,
                path=str(output_paths["review_sample"]),
                sample_size=review_sample_size,
            )

            if save_suspicious_debug:
                save_debug_suspicious_chunks(chunks, path=str(output_paths["suspicious"]))

            if save_sections_debug:
                lines = extract_pdf_lines(str(pdf_file))
                structured_items = build_structured_paragraphs(lines)
                save_debug_sections(structured_items, path=str(output_paths["sections_debug"]))

            chunks_total += len(chunks)
            files_ok += 1

            results.append({
                "status": "ok",
                "source_pdf": str(pdf_file),
                "chunks_file": str(output_paths["chunks"]),
                "report_file": str(output_paths["report"]),
                "review_sample_file": str(output_paths["review_sample"]),
                "chunks_total": len(chunks),
            })

            print(f"✅ {pdf_file} → {output_paths['chunks']} ({len(chunks)} chunks)")

        except Exception as exc:
            files_failed += 1
            results.append({
                "status": "error",
                "source_pdf": str(pdf_file),
                "error": str(exc),
            })

            print(f"❌ Ошибка при парсинге {pdf_file}: {exc}")

            if not continue_on_error:
                raise

    batch_report = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "recursive": recursive,
        "files_total": len(pdf_files),
        "files_ok": files_ok,
        "files_failed": files_failed,
        "chunks_total": chunks_total,
        "files": results,
    }

    save_report(batch_report, str(output_root / "batch_report.json"))
    return batch_report






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", help="PDF-файл или папка с PDF")
    parser.add_argument("--output", default="chunks.json")
    parser.add_argument(
        "--output-dir",
        default="chunks_by_file",
        help="Папка для режима --separate-files. Структура подпапок будет сохранена.",
    )
    parser.add_argument(
        "--separate-files",
        action="store_true",
        help="Парсить папку так, чтобы каждый PDF сохранился в отдельный *.chunks.json с сохранением структуры подпапок.",
    )
    parser.add_argument(
        "--chunks-suffix",
        default=".chunks.json",
        help="Суффикс для отдельных chunks-файлов в режиме --separate-files.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Если input_path — папка, искать PDF только в ней, без подпапок.",
    )

    parser.add_argument("--target-chars", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--min-chars", type=int, default=450)
    parser.add_argument("--overlap-chars", type=int, default=300)
    parser.add_argument("--max-page-span", type=int, default=2)
    parser.add_argument("--dedup-threshold", type=float, default=0.90)

    parser.add_argument("--document-title", default=None)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--specialty", default=None)
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Опционально записывает имя embedding-модели в каждый chunk. Сам парсер embeddings не считает.",
    )

    parser.add_argument("--debug-sections", action="store_true")
    parser.add_argument("--debug-suspicious", action="store_true")

    parser.add_argument("--report-output", default="chunks_report.json")
    parser.add_argument("--review-sample-output", default="chunks_review_sample.json")
    parser.add_argument("--review-sample-size", type=int, default=20)

    args = parser.parse_args()

    if args.separate_files:
        if args.document_title or args.document_id or args.topic or args.specialty:
            print(
                "⚠️  В режиме --separate-files параметры --document-title, --document-id, "
                "--topic и --specialty игнорируются. Для папки лучше заполнять DOCUMENT_CONFIG по именам PDF."
            )

        batch_report = parse_folder_to_separate_chunk_files(
            input_dir=args.input_path,
            output_dir=args.output_dir,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
            overlap_chars=args.overlap_chars,
            max_page_span=args.max_page_span,
            dedup_threshold=args.dedup_threshold,
            recursive=not args.no_recursive,
            save_sections_debug=args.debug_sections,
            save_suspicious_debug=args.debug_suspicious,
            review_sample_size=args.review_sample_size,
            chunks_suffix=args.chunks_suffix,
            embedding_model=args.embedding_model,
        )

        print(json.dumps(batch_report, ensure_ascii=False, indent=2))
        print(f"✅ Saved separate chunks files to {args.output_dir}")
        print(f"📊 Saved batch report to {Path(args.output_dir) / 'batch_report.json'}")

    else:
        chunks = parse_input_to_chunks(
            input_path=args.input_path,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
            overlap_chars=args.overlap_chars,
            max_page_span=args.max_page_span,
            dedup_threshold=args.dedup_threshold,
            recursive=not args.no_recursive,
            save_sections_debug=args.debug_sections,
            save_suspicious_debug=args.debug_suspicious,
            manual_document_title=args.document_title,
            topic=args.topic,
            specialty=args.specialty,
            manual_document_id=args.document_id,
            embedding_model=args.embedding_model,
        )

        save_chunks(chunks, args.output)

        report = build_chunks_report(chunks)
        save_report(report, args.report_output)

        save_random_review_sample(
            chunks,
            path=args.review_sample_output,
            sample_size=args.review_sample_size,
        )

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"✅ Saved {len(chunks)} knowledge chunks to {args.output}")
        print(f"📊 Saved report to {args.report_output}")
        print(f"🔎 Saved review sample to {args.review_sample_output}")
