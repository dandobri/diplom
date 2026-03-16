"""
Medical PDF Content Parser (Step 2)
====================================
Extracts the text content of each section, binding it to the structured
headings produced by heading_parser.py (Step 1).

Two-step workflow
-----------------
Step 1 — parse headings from TOC and save to JSON::

    python heading_parser.py document.pdf --json > document_headings.json

Step 2 — parse body content using the saved headings::

    python content_parser.py document.pdf --headings document_headings.json --json

If ``--headings`` is omitted, heading_parser is called automatically
(single-step mode, backward-compatible).

Strategy
--------
1. Load TOC headings (from file or by calling heading_parser directly).
2. Walk body pages using pdfplumber word-level font data.
3. Reconstruct lines with font metadata.
4. A body line is treated as a **section boundary** when its leading
   number matches an entry in the TOC whitelist (font is NOT required —
   the TOC is the authoritative source).  Each matched number is consumed
   so running page-headers cannot trigger duplicate sections.
5. Text between two consecutive section boundaries becomes the `content`
   of the preceding section.

Output
------
A list of section dicts::

    {
        "level":   "H1" | "H2" | "H3" | ...,
        "depth":   1 | 2 | 3 | ...,
        "number":  "1" | "1.1" | "2.3.1" | ...,
        "heading": "Full heading text from TOC",
        "content": "All body text under this heading until the next one",
    }
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required. Install it with: pip install pdfplumber")

# Import heading utilities from the same package
sys.path.insert(0, str(Path(__file__).parent))
from heading_parser import parse_headings, deduplicate, normalize_line as normalize_heading_line


# ─── Patterns ────────────────────────────────────────────────────────────────

HEADING_START_RE  = re.compile(r'^(\d+(?:\.\d+)*)\.*\s*(.*)', re.UNICODE)
# Matches ASCII dot-leaders and Unicode ellipsis leaders (… U+2026)
TOC_LEADER_RE     = re.compile(r'(?:\.{3,}|[\u2026…]{2,})')
PAGE_NUMBER_RE    = re.compile(r'^\d{1,4}$')
# Strips TOC leaders (ASCII or Unicode) + trailing page number
TOC_TRAILER_RE    = re.compile(r'[\s.\u2026…]{2,}\d*\s*$')


# ─── Font helpers ─────────────────────────────────────────────────────────────

def is_bold(fontname: str) -> bool:
    return 'bold' in fontname.lower()


def dominant_font(row: List[dict]) -> str:
    counts: Dict[str, int] = {}
    for w in row:
        counts[w['fontname']] = counts.get(w['fontname'], 0) + len(w['text'])
    return max(counts, key=lambda k: counts[k])


def avg_size(row: List[dict]) -> float:
    return sum(w['size'] for w in row) / len(row)


# ─── Page → Lines ─────────────────────────────────────────────────────────────

class BodyLine:
    __slots__ = ('text', 'bold', 'size')

    def __init__(self, text: str, bold: bool, size: float):
        self.text = text
        self.bold = bold
        self.size = size


def words_to_body_lines(page) -> List[BodyLine]:
    """Convert a pdfplumber page to BodyLine objects (grouped by y-coordinate)."""
    words = page.extract_words(extra_attrs=['fontname', 'size'])
    if not words:
        return []

    words.sort(key=lambda w: (round(w['top']), w['x0']))
    rows: List[List[dict]] = []
    cur: List[dict] = []
    last_top: float = -999.0

    for w in words:
        if abs(w['top'] - last_top) <= 2.0:
            cur.append(w)
        else:
            if cur:
                rows.append(cur)
            cur = [w]
            last_top = w['top']
    if cur:
        rows.append(cur)

    lines: List[BodyLine] = []
    for row in rows:
        # Sort within a row by x0 so left-to-right order is correct
        # (words on the same visual line may have slightly different top values
        # causing their global sort order to differ from their x position)
        row.sort(key=lambda w: w['x0'])
        text = ' '.join(w['text'] for w in row).strip()
        if not text:
            continue
        df = dominant_font(row)
        lines.append(BodyLine(
            text=text,
            bold=is_bold(df),
            size=avg_size(row),
        ))
    return lines


_TOC_HEADER_BODY_RE = re.compile(r'^\s*(Оглавление|Содержание)\s*$', re.UNICODE | re.IGNORECASE)
_PLAIN_ENTRY_BODY_RE = re.compile(r'^\d+[\.\s].*\s\d{1,4}\s*$', re.UNICODE)


def is_toc_page_body(lines: List[BodyLine]) -> bool:
    """True if this page is a TOC page (Format A: dot-leaders, or Format B: plain entries)."""
    texts = [ln.text for ln in lines]

    # Format A: ≥ 3 lines with dot/ellipsis leaders
    if sum(1 for t in texts if TOC_LEADER_RE.search(t)) >= 3:
        return True

    # Format B: page has "Оглавление"/"Содержание" header in first 5 lines
    if any(_TOC_HEADER_BODY_RE.match(t.strip()) for t in texts[:5]):
        return True

    # Format B fallback: ≥ 3 lines matching "<number> <text> <page_num>"
    if sum(1 for t in texts if _PLAIN_ENTRY_BODY_RE.match(t.strip())) >= 3:
        return True

    return False


# ─── Normalize heading number ─────────────────────────────────────────────────

def normalize_number(raw: str) -> str:
    """Normalize number extracted from a body heading line."""
    return raw.strip().rstrip('.')


# ─── Core: section extraction ─────────────────────────────────────────────────

def parse_sections(
    pdf_path: Union[str, Path],
    headings: Optional[List[dict]] = None,
) -> List[dict]:
    """Extract sections with heading + content from a medical PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    headings:
        Pre-parsed list of heading dicts (as produced by heading_parser).
        If *None*, heading_parser is called automatically.

    Returns a list of section dicts::

        {
            "level":   "H1" | "H2" | ...,
            "depth":   1 | 2 | ...,
            "number":  "1" | "1.1" | ...,
            "heading": "Heading text (from TOC)",
            "content": "Body text under this heading",
        }
    """
    pdf_path = Path(pdf_path)

    # ── Step 1: get TOC headings ────────────────────────────────────────────
    if headings is None:
        headings = deduplicate(parse_headings(pdf_path))
    # Build whitelist: number → heading dict (consumed as sections are found)
    heading_map: Dict[str, dict] = {h['number']: h for h in headings}
    # Ordered list of all heading numbers for out-of-order detection
    toc_numbers: List[str] = [h['number'] for h in headings]
    toc_pos_map: Dict[str, int] = {n: i for i, n in enumerate(toc_numbers)}
    # Pointer: index of the next expected heading (advances as headings are matched)
    current_toc_pos: int = 0

    # ── Step 2: walk body pages ─────────────────────────────────────────────
    # Strategy:
    #   • Bold line whose number is in the TOC whitelist → section boundary.
    #   • Bold lines immediately after a heading are skipped ONLY if they are
    #     fragments of the known TOC heading text (heading wraps/continuations).
    #   • Any bold line that is NOT a fragment of the heading text is real
    #     content (bold recommendations, bold terms, etc.) and is accumulated.
    # ── Helpers for text-based fallback matching ────────────────────────────
    def norm_heading_text(t: str) -> str:
        """Lowercase, collapse whitespace, strip punctuation for fuzzy comparison."""
        t = re.sub(r'[\s]+', ' ', t.strip()).lower()
        t = re.sub(r'[^\w\s]', '', t, flags=re.UNICODE)
        return t

    def find_heading_by_text(line_text_after_num: str) -> Optional[str]:
        """Return the number of the best-matching remaining heading, or None.

        Matches by checking if the normalised body text starts with (or equals)
        the normalised TOC heading text for headings whose depth matches the
        number depth from the body line.
        """
        norm_body = norm_heading_text(line_text_after_num)
        if not norm_body or len(norm_body) < 5:
            return None
        for num, h in heading_map.items():
            norm_toc = norm_heading_text(h['text'])
            if not norm_toc:
                continue
            # Accept if body text starts with TOC heading text (handles trailing garbage)
            if norm_body.startswith(norm_toc) or norm_toc.startswith(norm_body):
                return num
        return None

    sections: List[dict] = []
    current_section: Optional[dict] = None
    current_content_lines: List[str] = []
    heading_tail: str = ''  # normalised remaining heading text to match wraps against

    def flush_section() -> None:
        if current_section is not None:
            content = ' '.join(current_content_lines).strip()
            content = re.sub(r'\s+', ' ', content)
            current_section['content'] = content

    def strip_leaders(text: str) -> str:
        """Remove TOC dot-leaders (ASCII or Unicode) and trailing page number."""
        return TOC_TRAILER_RE.sub('', text).strip()

    def is_heading_fragment(line_text: str, toc_heading: str) -> bool:
        """Return True if line_text (after stripping leaders) is a fragment of toc_heading."""
        # Strip leaders/page numbers before comparing so wrap lines like
        # "состояний)…………7" correctly match against the clean heading text.
        clean_line = strip_leaders(line_text)
        norm_line = re.sub(r'\s+', ' ', clean_line).lower()
        norm_head = re.sub(r'\s+', ' ', toc_heading.strip()).lower()
        return bool(norm_line) and norm_line in norm_head

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            body_lines = words_to_body_lines(page)

            if is_toc_page_body(body_lines):
                continue

            for line in body_lines:
                stripped = normalize_heading_line(line.text.strip())

                if not stripped or PAGE_NUMBER_RE.match(stripped):
                    continue

                # Skip lines that look like TOC entries leaking into body pages
                # (wrap continuations with dot-leaders and a trailing page number)
                if TOC_LEADER_RE.search(stripped):
                    # Only skip if there's actual content before the leaders
                    # (i.e. it's a heading wrap, not body text with an ellipsis)
                    before_leaders = TOC_TRAILER_RE.sub('', stripped).strip()
                    if not before_leaders or PAGE_NUMBER_RE.match(before_leaders):
                        continue

                is_heading_font = line.bold or line.size >= 13.0

                # ── Check if this line is a section heading ─────────────────
                # A line is a section boundary when its leading number matches
                # the TOC whitelist.  The font check is used only for lines
                # whose number is NOT in the whitelist (prevents false positives
                # in plain body text while still allowing sub-headings that may
                # not be large/bold to be detected correctly).
                match = HEADING_START_RE.match(stripped)
                if match:
                    num = normalize_number(match.group(1))
                    body_text_after_num = match.group(2).strip()

                    # ── Text-based fallback ──────────────────────────────────
                    # If the number isn't in the whitelist but the line is bold/
                    # large AND its text matches a remaining heading by content,
                    # use that heading (handles PDF body numbering errors).
                    if num not in heading_map and is_heading_font and body_text_after_num:
                        fallback_num = find_heading_by_text(body_text_after_num)
                        if fallback_num is not None:
                            num = fallback_num

                    if num in heading_map:
                        # ── Out-of-order (running header) check ─────────────
                        # Running page-headers in these PDFs are always H1
                        # (top-level) headings printed at the top of each page.
                        # We only apply the out-of-order guard for depth-1
                        # headings: if there are still unmatched H1-or-higher
                        # headings in the TOC that should come BEFORE this one,
                        # this line is a running header → treat as body text.
                        # Sub-headings (H2, H3…) are NEVER skipped this way.
                        num_toc_pos = toc_pos_map.get(num, current_toc_pos)
                        candidate_depth = heading_map[num]['depth']
                        if candidate_depth == 1 and num_toc_pos > current_toc_pos:
                            is_running_header = any(
                                toc_numbers[i] in heading_map
                                for i in range(current_toc_pos, num_toc_pos)
                            )
                        else:
                            is_running_header = False

                        if is_running_header:
                            # Treat line as plain body content, not a boundary
                            if current_section is not None:
                                current_content_lines.append(stripped)
                            continue

                        flush_section()
                        if current_section is not None:
                            sections.append(current_section)

                        h = heading_map.pop(num)  # consume → prevents re-match
                        current_toc_pos = num_toc_pos + 1
                        current_section = {
                            'level':   h['level'],
                            'depth':   h['depth'],
                            'number':  num,
                            'heading': h['text'],
                            'content': '',
                        }
                        current_content_lines = []
                        # Store the full TOC heading text so we can
                        # recognise its wrap lines below.
                        heading_tail = h['text']
                        continue  # heading line itself is not content

                # ── Skip bold/large lines that are wrap fragments of the heading ───
                # If the current line is bold AND its text is a substring of
                # the known TOC heading text, it is a heading continuation
                # (line wrap) → skip.  Otherwise it is real content.
                if heading_tail and is_heading_font:
                    if is_heading_fragment(stripped, heading_tail):
                        continue  # heading wrap — skip

                # Once we pass the heading wraps, clear the tail so subsequent
                # bold lines are treated as content unconditionally.
                heading_tail = ''

                # ── Accumulate body text (bold emphasis or plain) ─────────
                if current_section is not None:
                    current_content_lines.append(stripped)

    # Flush last section
    flush_section()
    if current_section is not None:
        sections.append(current_section)

    return sections


# ─── Display helpers ──────────────────────────────────────────────────────────

def print_sections(sections: List[dict], content_preview: int = 120) -> None:
    """Pretty-print sections with a content preview."""
    indent_map = {1: '', 2: '  ', 3: '    ', 4: '      ', 5: '        '}
    for s in sections:
        prefix = indent_map.get(s['depth'], '          ')
        preview = s['content'][:content_preview].replace('\n', ' ')
        if len(s['content']) > content_preview:
            preview += '…'
        print(f"{prefix}[{s['level']}]  {s['number']}. {s['heading']}")
        if preview:
            print(f"{prefix}      ↳ {preview}")


def summary(sections: List[dict]) -> dict:
    """Return count of sections per level."""
    counts: dict = {}
    for s in sections:
        counts[s['level']] = counts.get(s['level'], 0) + 1
    return dict(sorted(counts.items()))


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python content_parser.py <path_to_pdf> [--headings headings.json] [--json] [--preview N]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: file not found — {pdf_path}")
        sys.exit(1)

    output_json = '--json' in sys.argv

    # Optional pre-parsed headings file
    headings: Optional[List[dict]] = None
    if '--headings' in sys.argv:
        idx = sys.argv.index('--headings')
        if idx + 1 < len(sys.argv):
            headings_path = Path(sys.argv[idx + 1])
            if not headings_path.exists():
                print(f"Error: headings file not found — {headings_path}")
                sys.exit(1)
            with open(headings_path, encoding='utf-8') as fh:
                headings = json.load(fh)

    # Optional preview length
    preview_len = 120
    if '--preview' in sys.argv:
        idx = sys.argv.index('--preview')
        if idx + 1 < len(sys.argv):
            try:
                preview_len = int(sys.argv[idx + 1])
            except ValueError:
                pass

    sections = parse_sections(pdf_path, headings=headings)

    if output_json:
        print(json.dumps(sections, ensure_ascii=False, indent=2))
    else:
        print(f"\n📄  Parsing sections: {pdf_path.name}")
        print("─" * 60)
        print_sections(sections, content_preview=preview_len)
        print()
        print("─" * 60)
        s = summary(sections)
        print("Summary:", s)
        print(f"Total sections: {sum(s.values())}")


if __name__ == '__main__':
    main()
