"""
PDF Parsing Pipeline
====================
Three-step processing for all PDFs in an input directory:

  Step 1: heading_parser  → <name>.json        (TOC headings)
  Step 2: content_parser  → <name>_content.json (sections with body text + lists)
  Step 3: table_parser    → tables added to each section

Usage
-----
  python pipeline.py <input_dir> <output_dir>

Example
-------
  python pipeline.py docs/test docs/test_parsed
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from heading_parser import parse_headings, deduplicate
from content_parser import parse_sections
from list_parser import annotate_sections as annotate_lists
from table_parser import annotate_sections as annotate_tables


def process_pdf(pdf_path: Path, out_dir: Path) -> None:
    name = pdf_path.stem

    # ── Step 1: parse headings from TOC ──────────────────────────────────────
    print(f"  [1/3] Parsing headings … ", end='', flush=True)
    headings = deduplicate(parse_headings(pdf_path))
    headings_out = out_dir / f"{name}.json"
    headings_out.write_text(
        json.dumps(headings, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"{len(headings)} headings → {headings_out.name}")

    # ── Step 2: parse body content + lists ──────────────────────────────────
    print(f"  [2/3] Parsing content  … ", end='', flush=True)
    sections = parse_sections(pdf_path, headings=headings)
    annotate_lists(sections)
    total_lists = sum(len(s.get('lists', [])) for s in sections)
    print(f"{len(sections)} sections, {total_lists} lists")

    # ── Step 3: extract tables ────────────────────────────────────────────────
    print(f"  [3/3] Extracting tables … ", end='', flush=True)
    annotate_tables(sections, pdf_path, headings=headings)
    total_tables = sum(len(s.get('tables', [])) for s in sections)

    content_out = out_dir / f"{name}_content.json"
    content_out.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"{total_tables} tables → {content_out.name}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir  = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(f"Error: input directory not found — {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(pdfs)} PDF(s) in {input_dir}\n")

    ok = 0
    for pdf in pdfs:
        print(f"📄 {pdf.name}")
        try:
            process_pdf(pdf, output_dir)
            ok += 1
        except Exception as exc:
            print(f"  ✗ ERROR: {exc}")
        print()

    print(f"Done: {ok}/{len(pdfs)} processed → {output_dir}")


if __name__ == '__main__':
    main()
