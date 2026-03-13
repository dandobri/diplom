"""
Medical PDF Heading Parser
==========================
Extracts structured headings (H1, H2, H3, …) from Russian medical PDF documents.

Strategy: TOC-first extraction
-------------------------------
The Table of Contents is the authoritative source for section headings.
We parse headings directly from the TOC pages.

Two TOC formats are supported:

  Format A — dot-leader style (classic):
      "1.1.Эпидемиология.............23"
      "2. Диагностика заболевания .....29"
    Detected when ≥ 3 lines on a page contain 3+ consecutive dots.

  Format B — plain style (modern):
      "1.1 Эпидемиология заболевания 9"
      "2. Диагностика заболевания или состояния, 15"
    Detected when a page starts with "Оглавление" or "Содержание" OR
    ≥ 3 lines match <number> <text> <trailing_integer> pattern.

Heading level is derived from numbering depth:
    "1."  or "1"   → H1
    "1.1" or "1.1."→ H2
    "2.3.1"        → H3
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from typing import List, Tuple, Union

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required. Install it with: pip install pdfplumber")


# ─── Patterns ────────────────────────────────────────────────────────────────

# Lines with 3+ consecutive dots → dot-leader (Format A)
TOC_LEADER_RE = re.compile(r'\.{3,}')

# Trailing dot-leaders + optional page number
TRAILING_DOTS_RE = re.compile(r'[\s.]{3,}\d*\s*$')

# Trailing standalone page number at end of line (Format B)
TRAILING_PAGE_RE = re.compile(r'\s+\d{1,4}\s*$')

# Page-number-only lines
PAGE_NUMBER_RE = re.compile(r'^\d{1,4}$')

# Matches a numbered heading after line normalization:
#   Group 1: the number (e.g. "1", "1.1", "2.3.1")
#   Group 2: the title text (may be merged with no separator)
# `\.*\s*` handles: trailing dot, space, or nothing (merged like "3.3Хирургическое")
HEADING_ENTRY_RE = re.compile(
    r'^(\d+(?:\.\d+)*)\.*\s*(.+)',
    re.UNICODE
)

# Section-header lines (not real headings)
SKIP_LINE_RE = re.compile(
    r'^(Оглавление|Содержание|Список\s+сокращений|Термины\s+и|'
    r'Приложени[еяя]|Критерии|Список\s+литературы)',
    re.UNICODE | re.IGNORECASE
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

# Pre-compiled pattern for space-broken numbers like "3. 3" or "1. 1. 2"
_SPACE_BROKEN_NUM_RE = re.compile(r'^(\d+(?:\.\s*\d+)*)\.\s+(\d)')

def normalize_line(line: str) -> str:
    """Fix space-broken heading numbers at the start of a line.

    Some PDFs encode "3.3 Хирургическое" as "3. 3Хирургическое" or
    "3. 3 Хирургическое" (space between dot and next digit).

    Examples:
        "3. 3Хирургическое"     → "3.3 Хирургическое"
        "1. 4. 2 Диагностика"   → "1.4.2 Диагностика"
    """
    m = _SPACE_BROKEN_NUM_RE.match(line)
    if not m:
        return line
    # Rebuild: collapse all internal spaces/dots, then append rest
    prefix = m.group(1)   # e.g. "3"
    next_d = m.group(2)   # e.g. "3"
    new_num = re.sub(r'\.\s*', '.', prefix) + '.' + next_d
    rest    = line[m.end():]   # rest of line after the next digit
    return new_num + rest


def normalize_number(raw: str) -> str:
    """Normalize a heading number, removing spaces between segments.

    Examples:
        "1.5"   → "1.5"   (already clean)
        "1.1."  → "1.1"
        "2"     → "2"
    """
    normalized = raw.strip().rstrip('.')
    return normalized


def heading_level(numbering: str) -> int:
    """Return heading depth (1 = H1) from dot-separated numbering string."""
    return len(numbering.split('.'))


def clean_text(text: str) -> str:
    """Collapse multiple spaces / normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def strip_toc_trailer_A(text: str) -> str:
    """Strip trailing dot-leaders + page number (Format A).

    Handles both 3+ dots ("...29") and 2-dot variants ("..20").
    """
    return re.sub(r'[\s.]{2,}\d*\s*$', '', text).strip()


def strip_toc_trailer_B(text: str) -> str:
    """Strip trailing standalone page number (Format B)."""
    return re.sub(r'\s+\d{1,4}\s*$', '', text).strip()


# ─── TOC page detection ───────────────────────────────────────────────────────

def classify_toc_page(lines: List[str]) -> str:
    """Return 'A', 'B', or '' indicating the TOC format of this page.

    'A' = dot-leader format
    'B' = plain (trailing page number) format
    ''  = not a TOC page
    """
    # Format A: ≥ 3 lines have dot-leaders
    dot_leader_count = sum(1 for ln in lines if TOC_LEADER_RE.search(ln.strip()))
    if dot_leader_count >= 3:
        return 'A'

    # Format B: page starts with "Оглавление"/"Содержание" OR
    # ≥ 3 lines match: number at start, digit at end
    toc_header = any(
        re.match(r'^\s*(Оглавление|Содержание)', ln, re.IGNORECASE)
        for ln in lines[:5]
    )
    plain_entry_re = re.compile(r'^\d+[\.\s].*\s\d{1,4}\s*$', re.UNICODE)
    plain_entry_count = sum(1 for ln in lines if plain_entry_re.match(ln.strip()))

    if toc_header or plain_entry_count >= 3:
        return 'B'

    return ''


# ─── TOC line parsing ─────────────────────────────────────────────────────────

def parse_toc_line(stripped: str, fmt: str) -> Tuple[str, str]:
    """Parse a single TOC line; return (normalized_number, title_text).

    Returns ('', '') if the line is not a valid heading entry.
    """
    if SKIP_LINE_RE.match(stripped):
        return ('', '')
    if PAGE_NUMBER_RE.match(stripped):
        return ('', '')

    match = HEADING_ENTRY_RE.match(stripped)
    if not match:
        return ('', '')

    raw_number = match.group(1)
    rest       = match.group(2)

    # Normalize the number (handle spaces like "1. 5" → "1.5")
    number = normalize_number(raw_number)

    # Strip trailing page reference from title
    if fmt == 'A':
        title = strip_toc_trailer_A(rest)
    else:
        title = strip_toc_trailer_B(rest)

    # After stripping, if title is empty or is just a number, skip
    if not title or PAGE_NUMBER_RE.match(title):
        return ('', '')

    return (number, title)


# ─── Core extraction ─────────────────────────────────────────────────────────

def extract_text_pages(pdf_path: Union[str, Path]) -> List[List[str]]:
    """Return text lines grouped by page."""
    pages: List[List[str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            raw = page.extract_text()
            pages.append(raw.split('\n') if raw else [])
    return pages


def parse_headings(pdf_path: Union[str, Path]) -> List[dict]:
    """Parse structured headings from the TOC of a medical PDF.

    Returns a list of dicts::

        {
            "level":   "H1" | "H2" | "H3" | ...,
            "depth":   1 | 2 | 3 | ...,
            "number":  "1" | "1.1" | "2.3.1" | ...,
            "text":    "Full heading text",
        }
    """
    pages = extract_text_pages(pdf_path)
    headings: List[dict] = []

    for page_lines in pages:
        fmt = classify_toc_page(page_lines)
        if not fmt:
            continue

        i = 0
        while i < len(page_lines):
            stripped = normalize_line(page_lines[i].strip())

            if not stripped:
                i += 1
                continue

            number, title = parse_toc_line(stripped, fmt)
            if not number:
                i += 1
                continue

            # ── Collect wrapped continuation lines ──────────────────────────
            # TOC multi-line entries look like:
            #   Line 1: "1. Краткая информация по заболеванию или состоянию (группе заболеваний или"
            #   Line 2: "состояний) ............6"   ← has dot-leaders but is still part of heading
            #
            # Rule:
            #   - If a continuation line has NO dot-leaders and does NOT start a
            #     new heading number → collect it and continue.
            #   - If a continuation line HAS dot-leaders → it is the LAST wrapped
            #     line of the current heading.  Strip its trailer, collect the
            #     remaining text, then STOP.
            j = i + 1
            while j < len(page_lines):
                nxt = page_lines[j].strip()

                if not nxt or PAGE_NUMBER_RE.match(nxt):
                    break
                if SKIP_LINE_RE.match(nxt):
                    break

                # Does it start a new heading entry?
                nxt_num, _ = parse_toc_line(nxt, fmt)
                if nxt_num:
                    break

                if TOC_LEADER_RE.search(nxt):
                    # Final wrapped line — grab text before the dot-leaders
                    if fmt == 'A':
                        cont = strip_toc_trailer_A(nxt)
                    else:
                        cont = strip_toc_trailer_B(nxt)
                    if cont:
                        title = title + ' ' + cont
                    j += 1
                    break  # heading ends here

                # Plain continuation line (no dot-leaders)
                if fmt == 'A':
                    cont = strip_toc_trailer_A(nxt)
                else:
                    cont = strip_toc_trailer_B(nxt)

                if cont:
                    title = title + ' ' + cont
                j += 1

            full_text = clean_text(title)
            if not full_text:
                i = j
                continue

            depth = heading_level(number)
            headings.append({
                'level':  f'H{depth}',
                'depth':  depth,
                'number': number,
                'text':   full_text,
            })

            i = j

    return headings


# ─── Deduplication ───────────────────────────────────────────────────────────

def deduplicate(headings: List[dict]) -> List[dict]:
    """Remove duplicate headings (same number + text)."""
    seen = set()
    result = []
    for h in headings:
        key = (h['number'], h['text'])
        if key not in seen:
            seen.add(key)
            result.append(h)
    return result


# ─── Display helpers ──────────────────────────────────────────────────────────

def print_headings(headings: List[dict]) -> None:
    """Pretty-print with visual indentation."""
    indent_map = {1: '', 2: '  ', 3: '    ', 4: '      ', 5: '        '}
    for h in headings:
        prefix = indent_map.get(h['depth'], '          ')
        print(f"{prefix}[{h['level']}]  {h['number']}. {h['text']}")


def summary(headings: List[dict]) -> dict:
    """Return count of headings per level."""
    counts: dict = {}
    for h in headings:
        counts[h['level']] = counts.get(h['level'], 0) + 1
    return dict(sorted(counts.items()))


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python heading_parser.py <path_to_pdf> [--json] [--no-dedup]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: file not found — {pdf_path}")
        sys.exit(1)

    output_json = '--json'     in sys.argv
    no_dedup    = '--no-dedup' in sys.argv

    headings = parse_headings(pdf_path)
    if not no_dedup:
        headings = deduplicate(headings)

    if output_json:
        # Pure JSON — no decorative output so the result is valid JSON
        print(json.dumps(headings, ensure_ascii=False, indent=2))
    else:
        print(f"\n📄  Parsing: {pdf_path.name}")
        print("─" * 60)
        print_headings(headings)
        print()
        print("─" * 60)
        s = summary(headings)
        print("Summary:", s)
        print(f"Total headings found: {sum(s.values())}")


if __name__ == '__main__':
    main()
