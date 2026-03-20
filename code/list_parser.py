"""
List Parser
===========
Extracts structured lists (bullet and numbered) from a section content string,
and classifies each list with a ``topic`` label derived from the section heading.

The content is a flat string produced by content_parser.py. List items are
identified by their markers and extracted into structured objects.

Supported list types
--------------------
- **bullet**   — markers: •  ●  ○  ▪  ■  ►
                 OR a leading dash (- / – / —) when it functions as a list
                 marker (i.e. at least half the resulting items end with ;
                 or the group starts right after ;).
- **numbered** — markers: "1) 2) 3)" or "1. 2. 3." (1–2 digits, then ) or .)
                 followed by a Cyrillic/Latin letter (to avoid decimal numbers
                 or reference numbers like "[3]").
                 Citation-style lists (items matching author/year patterns)
                 are suppressed.

Output format
-------------
Each section gets a ``lists`` field::

    [
        {
            "type":  "bullet" | "numbered",
            "topic": "симптомы" | "диагностика" | "лечение" | ...,
            "items": ["item text 1", "item text 2", ...]
        },
        ...
    ]

Topic values
------------
Based on the section heading (see ``classify_topic``):
  "определение", "этиология", "эпидемиология", "классификация",
  "симптомы", "диагностика", "лабораторная_диагностика",
  "инструментальная_диагностика", "лечение", "хирургическое_лечение",
  "реабилитация", "профилактика", "организация", "прочее"

Multiple distinct lists within a section are returned as separate objects.
An empty list [] is returned when no lists are found.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# ─── Patterns ────────────────────────────────────────────────────────────────

# Unicode bullet markers (reliable — never appear mid-word)
UNICODE_BULLET_RE = re.compile(
    r'(?:^|(?<=[\s;.]))\s*[•●○▪■►]\s*',
    re.UNICODE | re.MULTILINE
)

# Dash-as-list-marker: -, –, — preceded by whitespace or start-of-string,
# followed by whitespace + non-digit letter. Used only when it actually forms
# a list (see _is_dash_list() below).
DASH_MARKER_RE = re.compile(
    r'(?:^|(?<=\s))[-–—]\s+(?=[А-ЯЁA-Zа-яёa-z])',
    re.UNICODE | re.MULTILINE
)

# Numbered markers: 1-2 digits + ) or . + whitespace + letter
NUMBERED_MARKER_RE = re.compile(
    r'(?:^|(?<=\s))\d{1,2}[)]\s+(?=[А-ЯЁA-Zа-яёa-z])'  # only "1)" style
    r'|(?:^|(?<=\s))\d{1,2}[.]\s+(?=[А-ЯЁа-яё])',        # "1." only before Cyrillic
    re.UNICODE | re.MULTILINE
)

# Citation / reference heuristic: item looks like a bibliography entry
_CITATION_RE = re.compile(
    r'(?:\d{4}[;.,]|Vol\.|et al|// |pp?\.\s*\d|doi:|ISBN)',
    re.IGNORECASE
)

# Minimum item length
MIN_ITEM_LEN = 8

# Minimum number of items to call it a list
MIN_LIST_SIZE = 2

# ─── Topic classification ─────────────────────────────────────────────────────

# Each entry: (topic_label, list_of_keyword_patterns)
# Patterns are matched case-insensitively against the section heading.
_TOPIC_RULES: List[Tuple[str, List[str]]] = [
    ('определение',              ['определение']),
    ('этиология',                ['этиология', 'патогенез']),
    ('эпидемиология',            ['эпидемиология']),
    ('кодирование',              ['кодирован', 'мкб', 'классификац.*болезн']),
    ('классификация',            ['классификац']),
    ('симптомы',                 ['клиническ.*картин', 'симптом', 'жалоб', 'анамнез']),
    ('лабораторная_диагностика', ['лабораторн']),
    ('инструментальная_диагностика', ['инструментальн']),
    ('диагностика',              ['диагностик', 'обследован', 'физикальн', 'иные диагностическ']),
    ('хирургическое_лечение',    ['хирургическ']),
    ('лечение',                  ['лечени', 'терапи', 'медикаментозн', 'консерватив', 'немедикаментозн']),
    ('реабилитация',             ['реабилитац', 'санаторн']),
    ('профилактика',             ['профилактик', 'диспансерн']),
    ('организация',              ['организац.*помощ', 'оказани.*помощ']),
    ('дополнительная_информация', ['дополнительн.*информац', 'факторы.*исход']),
    ('осложнения',               ['осложнени']),
    ('прогноз',                  ['прогноз', 'исход']),
]

_COMPILED_RULES: List[Tuple[str, List[re.Pattern]]] = [
    (label, [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns])
    for label, patterns in _TOPIC_RULES
]


def classify_topic(heading: str) -> str:
    """Return a topic label for a list based on the section heading text."""
    if not heading:
        return 'прочее'
    for label, patterns in _COMPILED_RULES:
        if any(p.search(heading) for p in patterns):
            return label
    return 'прочее'


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clean_item(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r';\s*$', '', text).strip()
    return text


def _split_on_pattern(content: str, pattern: re.Pattern) -> List[str]:
    """Split content at all match positions; return text segments after each match."""
    matches = list(pattern.finditer(content))
    if not matches:
        return []
    segments = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        segments.append(content[start:end])
    return segments


def _is_dash_list(segments: List[str]) -> bool:
    """Return True if dash-split segments look like a real list.

    Heuristic: ≥ 40% of items end with ';' — the defining characteristic
    of a proper dash-separated list in Russian medical documents.
    Short items alone are NOT sufficient because dashes are also used as
    em-dashes in definitions ("Холецистит – воспаление...").
    """
    if len(segments) < MIN_LIST_SIZE:
        return False
    semicolon_count = sum(1 for s in segments if s.rstrip().endswith(';'))
    return semicolon_count >= max(1, len(segments) * 0.4)


def _is_citation_list(items: List[str]) -> bool:
    """Return True if the majority of items look like bibliography citations."""
    if not items:
        return False
    citation_count = sum(1 for it in items if _CITATION_RE.search(it))
    return citation_count >= len(items) * 0.4


# ─── Core ────────────────────────────────────────────────────────────────────

def extract_lists(content: str, heading: str = '') -> List[dict]:
    """Extract bullet and numbered lists from a flat content string.

    Parameters
    ----------
    content:
        The section body text.
    heading:
        The section heading (used to classify the topic of each list).
    """
    if not content or not content.strip():
        return []

    topic = classify_topic(heading)
    result: List[dict] = []

    # ── Unicode bullet lists ──────────────────────────────────────────────────
    ub_segments = _split_on_pattern(content, UNICODE_BULLET_RE)
    ub_items = [_clean_item(s) for s in ub_segments]
    ub_items = [it for it in ub_items if len(it) >= MIN_ITEM_LEN]
    if len(ub_items) >= MIN_LIST_SIZE and not _is_citation_list(ub_items):
        result.append({"type": "bullet", "topic": topic, "items": ub_items})

    # ── Dash bullet lists ─────────────────────────────────────────────────────
    dash_segments = _split_on_pattern(content, DASH_MARKER_RE)
    if _is_dash_list(dash_segments):
        dash_items = [_clean_item(s) for s in dash_segments]
        dash_items = [it for it in dash_items if len(it) >= MIN_ITEM_LEN]
        if len(dash_items) >= MIN_LIST_SIZE and not _is_citation_list(dash_items):
            if not ub_items or set(dash_items) != set(ub_items):
                result.append({"type": "bullet", "topic": topic, "items": dash_items})

    # ── Numbered lists ────────────────────────────────────────────────────────
    num_segments = _split_on_pattern(content, NUMBERED_MARKER_RE)
    num_items = [_clean_item(s) for s in num_segments]
    num_items = [it for it in num_items if len(it) >= MIN_ITEM_LEN]
    if len(num_items) >= MIN_LIST_SIZE and not _is_citation_list(num_items):
        result.append({"type": "numbered", "topic": topic, "items": num_items})

    return result


def annotate_sections(sections: List[dict]) -> List[dict]:
    """Add a ``lists`` field to each section dict in-place and return the list."""
    for section in sections:
        section['lists'] = extract_lists(
            section.get('content') or '',
            heading=section.get('heading') or '',
        )
    return sections


# ─── CLI / demo ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import json
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python list_parser.py <content_json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    sections = json.loads(path.read_text(encoding='utf-8'))
    annotate_sections(sections)

    total_lists = sum(len(s['lists']) for s in sections)
    total_items = sum(len(lst['items']) for s in sections for lst in s['lists'])
    print(f"Found {total_lists} lists, {total_items} items total\n")

    for s in sections:
        if s['lists']:
            print(f"[{s['level']}] {s['number']}. {s['heading'][:60]}")
            for lst in s['lists']:
                print(f"  type={lst['type']}  ({len(lst['items'])} items)")
                for item in lst['items'][:4]:
                    print(f"    • {item[:90]}")
                if len(lst['items']) > 4:
                    print(f"    … +{len(lst['items'])-4} more")
            print()
