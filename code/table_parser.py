"""
table_parser.py
===============
Extract tables from Russian medical clinical-guideline PDFs and attach each
table to the correct section (as identified by content_parser).

Public API
----------
annotate_sections(sections, pdf_path) -> List[dict]
    Adds a ``tables`` key to each section dict.  Each entry in ``tables`` is::

        {
            "caption": str | null,   # text line immediately before the table
            "rows":    [[cell, ...], ...]  # 2-D array of cleaned strings
        }

    Tables considered non-diagnostic are silently dropped:
    - quality checklists  (№ | Критерии качества | Да/нет)
    - evidence/UUR legend tables
    - layout artefacts    (1-row, or all cells empty)
    - appendix tables     (pdfplumber detects no real table structure → ignored)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover
    raise ImportError("pdfplumber is required: pip install pdfplumber") from exc

from heading_parser import parse_headings, deduplicate, normalize_line as normalize_heading_line
from content_parser import words_to_body_lines, is_toc_page_body

# ── Heading-number pattern (same as content_parser) ──────────────────────────
_HEADING_NUM_RE = re.compile(
    r'^(\d+(?:\.\d+)*)\s*\.?\s+(.*)$',
    re.DOTALL,
)

# ── Table-type detection ──────────────────────────────────────────────────────

def _flat_text(table: List[List]) -> str:
    """Flatten all cells of a table into a single lower-case string."""
    return ' '.join(
        str(cell or '').replace('\n', ' ').strip()
        for row in table
        for cell in row
    ).lower()


def _first_cells(table: List[List]) -> List[str]:
    """Return text of all cells in the first non-empty row."""
    for row in table:
        cells = [str(c or '').replace('\n', ' ').strip() for c in row]
        if any(cells):
            return cells
    return []


_HYPHEN_WRAP_RE = re.compile(r'-\s+')


def _normalize_flat(text: str) -> str:
    """Remove soft-hyphen line-breaks so 'убедитель-\nности' → 'убедительности'."""
    return _HYPHEN_WRAP_RE.sub('', text)


def _is_quality_checklist(table: List[List]) -> bool:
    """True if table is a quality-indicator checklist or an author/appendix table."""
    raw_flat = _flat_text(table)
    flat = _normalize_flat(raw_flat)

    # Standard quality checklist with да/нет column
    if 'да/нет' in flat or 'да / нет' in flat or '(да/нет)' in flat:
        return True
    # Quality checklist that uses УУР/УДД level columns instead of да/нет
    if 'критерии качества' in flat and (
        'убедительности' in flat or 'достоверности' in flat
        or 'уур' in flat
    ):
        return True
    if 'критерии качества' in flat and 'выполнения' in flat:
        return True
    # Author / working group roster tables
    if ('рабочей группы' in flat or 'рабочая группа' in flat or
            'члены рабочей' in flat):
        return True
    # Personal academic credentials → author roster
    if (re.search(r'к\.м\.н\.|д\.м\.н\.|кандидат медицинских|доктор медицинских', flat) and
            re.search(r'\bминздрав\b|\bфгбу\b|\bгбу\b', flat)):
        return True
    return False


def _is_evidence_legend(table: List[List]) -> bool:
    """True if table is an evidence-level legend (УУР/УДД decoding table)."""
    flat = _normalize_flat(_flat_text(table))
    cells = _first_cells(table)
    first = cells[0].lower().strip() if cells else ''
    if first in ('ууr', 'уур', 'удд'):
        return True
    if 'сильная рекомендация' in flat and 'уур' in flat:
        return True
    # Fragment lines containing only recommendation-level text
    if 'уровень убедительности рекомендаций' in flat:
        return True
    # Numbered evidence-level rows (4 | Несравнительные исследования...)
    if 'несравнительные исследования' in flat or 'описание клинического случая' in flat:
        return True
    if 'обоснование механизма действия' in flat:
        return True
    # УДД legend table starting with number rows
    if re.match(r'^[1-5]$', first) and (
        'систематический обзор' in flat or 'рандомизированн' in flat
    ):
        return True
    return False


def _is_layout_artifact(table: List[List]) -> bool:
    """True if the table has no useful structure (≤2 rows, or almost all cells empty)."""
    if not table:
        return True
    # Tables with ≤2 rows are almost always running headers, footnotes, or sentence fragments
    if len(table) <= 2:
        return True
    # Count non-empty cells
    total = sum(len(row) for row in table)
    nonempty = sum(
        1 for row in table for c in row
        if str(c or '').strip()
    )
    if total > 0 and nonempty / total < 0.15:
        return True
    return False


def _is_body_text_wrap(table: List[List]) -> bool:
    """True if the table is body text laid out in one dominant column.

    Handles three cases:
    • 2-col table where first col dominates (standard left-flow layout)
    • 2-col table where second col dominates (right-flow layout)
    • Any table where a single column contains ≥70 % of all non-empty cells
    """
    if not table:
        return False
    ncols = max(len(row) for row in table)
    if ncols < 2:
        return False

    # Count non-empty cells per column.
    # NOTE: use (row[ci] or '') to avoid str(None) == 'None' being counted as content.
    col_nonempty = [0] * ncols
    total_nonempty = 0
    for row in table:
        for ci in range(ncols):
            raw = row[ci] if ci < len(row) else None
            val = str(raw or '').strip()
            if val:
                col_nonempty[ci] += 1
                total_nonempty += 1

    if total_nonempty == 0:
        return True  # all empty → artefact

    dominant = max(col_nonempty)
    # If one column has ≥70 % of all content, it's a text-flow layout artefact
    return dominant / total_nonempty >= 0.70


_DIAGNOSTIC_HEADER_RE = re.compile(
    r'крит(?:ерии|ерий|ериев)|'
    r'диагноз\w*|'
    r'классификаци\w+|'
    r'степен(?:ь|и)\s+тяжест\w+|'
    r'шкал\w+\s+(?:оценки|диагностики|тяжести)|'
    r'признак\w+\s+(?:диагноза|воспалени\w+)|'
    r'дифференциальный\s+диагноз',
    re.IGNORECASE | re.UNICODE,
)


def _is_diagnostic_header(text: str) -> bool:
    """True if text looks like a diagnostic-criteria section header."""
    return bool(_DIAGNOSTIC_HEADER_RE.search(text))


def _should_skip(table: List[List]) -> bool:
    return (
        _is_layout_artifact(table) or
        _is_quality_checklist(table) or
        _is_evidence_legend(table) or
        _is_body_text_wrap(table)
    )


# ── Cell cleaning ─────────────────────────────────────────────────────────────

def _clean_cell(text) -> str:
    if text is None:
        return ''
    s = str(text).replace('\n', ' ')
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()


def _clean_table(table: List[List]) -> List[List[str]]:
    return [[_clean_cell(c) for c in row] for row in table]


# ── Caption detection ─────────────────────────────────────────────────────────

_CAPTION_RE = re.compile(
    r'(?:таблица|табл\.?|table)\s*\d*',
    re.IGNORECASE | re.UNICODE,
)


def _find_caption(page, table_bbox: Tuple) -> Optional[str]:
    """
    Look for a 'Таблица N' caption line immediately above the table bounding box.

    Strategy: extract all words on the page whose top coordinate is within
    ~20pt above the table top, assemble them left-to-right, and check if the
    result looks like a caption.
    """
    t_x0, t_top, t_x1, t_bottom = table_bbox
    gap = 22  # pt above table top to search
    candidates = []

    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    for w in words:
        w_top = w.get('top', 0)
        w_bottom = w.get('bottom', 0)
        if t_top - gap <= w_top <= t_top and w_bottom <= t_top + 4:
            candidates.append(w)

    if not candidates:
        return None

    candidates.sort(key=lambda w: (round(w['top']), w['x0']))
    line = ' '.join(w['text'] for w in candidates).strip()
    if _CAPTION_RE.search(line):
        return line
    return None


# ── Section tracking ──────────────────────────────────────────────────────────

def _build_page_section_map(
    pdf_path: Path,
    headings: List[dict],
) -> Dict[int, str]:
    """
    Return a mapping  {page_index_0based → section_number}  by re-scanning
    body pages with the same heading-detection logic as content_parser.

    The section_number for a page is the *last heading* seen on that page or
    any earlier page (i.e. the currently-open section).
    """
    heading_map: Dict[str, dict] = {h['number']: h for h in headings}
    toc_numbers: List[str] = [h['number'] for h in headings]
    toc_pos_map: Dict[str, int] = {n: i for i, n in enumerate(toc_numbers)}
    current_toc_pos: int = 0

    page_section: Dict[int, str] = {}
    current_section_num: Optional[str] = None

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            body_lines = words_to_body_lines(page)

            if is_toc_page_body(body_lines):
                page_section[page_idx] = current_section_num
                continue

            for bl in body_lines:
                if not bl.bold:
                    continue
                norm_text = normalize_heading_line(bl.text)
                m = _HEADING_NUM_RE.match(norm_text)
                if not m:
                    continue
                num = m.group(1)
                if num not in heading_map:
                    continue
                num_toc_pos = toc_pos_map.get(num, -1)

                # Out-of-order running-header guard (depth-1 only)
                depth = num.count('.') + 1
                if depth == 1 and num_toc_pos > current_toc_pos:
                    is_running = any(
                        toc_numbers[i] in heading_map
                        for i in range(current_toc_pos, num_toc_pos)
                    )
                    if is_running:
                        continue

                # Consume heading
                heading_map.pop(num, None)
                current_section_num = num
                if num_toc_pos >= 0:
                    current_toc_pos = num_toc_pos + 1

            page_section[page_idx] = current_section_num

    return page_section


# ── Main public function ──────────────────────────────────────────────────────

def annotate_sections(
    sections: List[dict],
    pdf_path: Union[str, Path],
    headings: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Extract tables from *pdf_path* and attach each table to the matching
    section in *sections*.

    Each section gets a new key ``"tables"`` (possibly empty list).

    Parameters
    ----------
    sections:
        List of section dicts as produced by content_parser.
    pdf_path:
        Path to the source PDF.
    headings:
        Pre-parsed headings (optional; auto-parsed if omitted).
    """
    pdf_path = Path(pdf_path)

    if headings is None:
        headings = deduplicate(parse_headings(pdf_path))

    # Index sections by number for O(1) lookup
    section_by_num: Dict[str, dict] = {s['number']: s for s in sections}

    # Initialise tables list for every section
    for s in sections:
        s.setdefault('tables', [])

    if not headings:
        return sections

    # Build page → section_number map
    page_section = _build_page_section_map(pdf_path, list(headings))  # pass a copy

    # Walk pages, extract tables, filter and attach
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            sec_num = page_section.get(page_idx)
            if sec_num is None:
                continue
            section = section_by_num.get(sec_num)
            if section is None:
                continue

            raw_tables = page.find_tables()
            extracted = page.extract_tables()

            for tbl_obj, tbl_data in zip(raw_tables, extracted):
                if _should_skip(tbl_data):
                    continue

                caption = _find_caption(page, tbl_obj.bbox)
                cleaned = _clean_table(tbl_data)

                max_cols = max((len(r) for r in cleaned), default=0)

                if max_cols < 2:
                    # Allow single-column tables ONLY if the first cell looks
                    # like a diagnostic-criteria heading.
                    first_row_text = cleaned[0][0] if cleaned else ''
                    if not _is_diagnostic_header(first_row_text):
                        continue

                section['tables'].append({
                    'caption': caption,
                    'rows': cleaned,
                })

    return sections


# ── CLI (for manual inspection) ───────────────────────────────────────────────

def main() -> None:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description='Extract tables from a medical PDF.')
    ap.add_argument('pdf', help='Path to the PDF file')
    ap.add_argument('--headings', help='Pre-parsed headings JSON (optional)')
    ap.add_argument('--content', help='Pre-parsed sections JSON (optional)')
    ap.add_argument('--out', help='Write output JSON to this path (default: stdout)')
    args = ap.parse_args()

    if args.headings:
        with open(args.headings, encoding='utf-8') as f:
            headings = json.load(f)
    else:
        headings = None

    if args.content:
        with open(args.content, encoding='utf-8') as f:
            sections = json.load(f)
    else:
        from content_parser import parse_sections
        sections = parse_sections(args.pdf, headings=headings)

    sections = annotate_sections(sections, args.pdf, headings=headings)

    total_tables = sum(len(s.get('tables', [])) for s in sections)
    print(f'  {total_tables} tables extracted across {len(sections)} sections',
          file=sys.stderr)

    out_json = json.dumps(sections, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_json, encoding='utf-8')
        print(f'  Written to {args.out}', file=sys.stderr)
    else:
        print(out_json)


if __name__ == '__main__':
    main()
