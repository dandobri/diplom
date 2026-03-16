"""
Parser Test Suite
=================
Validates structural correctness and content placement for all parsed PDFs.

Usage
-----
  python test_parser.py                   # run all tests
  python test_parser.py --verbose         # show details for every check

What is tested
--------------
Structural (automatic, applied to every file):
  1. No duplicate section numbers in content output.
  2. Sections appear in the same order as in the TOC.
  3. No TOC dot-leaders / page numbers leak into section headings or content.
  4. Every section has a non-None content field.
  5. Heading numbers in content match headings file.

Content spot-checks (per-file, against known phrases that must appear
in a specific section):
  - A phrase that is unmistakably part of a section is checked to be
    present in that section's content (or its children).
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Helpers ──────────────────────────────────────────────────────────────────

LEADER_RE = re.compile(r'[\u2026…]{2,}|\.{3,}')

def load(name: str, suffix: str) -> Optional[list]:
    path = Path(f'docs/test_parsed/{name}{suffix}.json')
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def section_map(content: list) -> Dict[str, dict]:
    return {s['number']: s for s in content}


# ─── Structural checks ────────────────────────────────────────────────────────

def check_no_duplicates(content: list) -> List[str]:
    errors = []
    seen = {}
    for s in content:
        n = s['number']
        if n in seen:
            errors.append(f"Duplicate section number '{n}'")
        seen[n] = True
    return errors


def check_order(headings: list, content: list) -> List[str]:
    """Sections in content must follow TOC order."""
    errors = []
    toc_pos = {h['number']: i for i, h in enumerate(headings)}
    prev_pos = -1
    for s in content:
        pos = toc_pos.get(s['number'], -1)
        if pos == -1:
            continue  # not in TOC (shouldn't happen after other checks)
        if pos < prev_pos:
            errors.append(
                f"Section '{s['number']}' appears out of TOC order"
            )
        prev_pos = pos
    return errors


def check_no_leader_bleed(content: list) -> List[str]:
    """No section heading or content should start with dot-leaders / ellipsis."""
    errors = []
    for s in content:
        heading = s.get('heading', '')
        c = s.get('content', '') or ''
        if LEADER_RE.search(heading):
            errors.append(
                f"Section '{s['number']}' heading contains leaders: {heading[:60]!r}"
            )
        if c and LEADER_RE.match(c.strip()):
            errors.append(
                f"Section '{s['number']}' content starts with leaders: {c[:60]!r}"
            )
    return errors


def check_content_not_none(content: list) -> List[str]:
    errors = []
    for s in content:
        if s.get('content') is None:
            errors.append(f"Section '{s['number']}' has None content")
    return errors


def check_numbers_match_toc(headings: list, content: list) -> List[str]:
    errors = []
    toc_nums = {h['number'] for h in headings}
    for s in content:
        if s['number'] not in toc_nums:
            errors.append(
                f"Section '{s['number']}' not found in TOC"
            )
    return errors


# ─── Content spot-checks ──────────────────────────────────────────────────────
# Format: (file_stem, section_number, phrase_that_must_appear_in_content)
# The phrase is checked against the section itself AND all its direct children
# (so H1 checks also pass if phrase is in an H2 child).

SPOT_CHECKS: List[Tuple[str, str, str]] = [
    # КР246_3 — Экзема
    ('КР246_3', '1.1', 'Экзема'),
    ('КР246_3', '1.2', 'патогенез'),
    ('КР246_3', '1.3', 'распространенност'),
    ('КР246_3', '3.1', 'пимекролимус'),

    # КР25_2 — ОРВИ у детей
    ('КР25_2',  '1.1', 'ОРВИ'),
    ('КР25_2',  '1.3', 'заболеваемость'),
    ('КР25_2',  '3.1', 'Осельтамивир'),
    ('КР25_2',  '5',   'вакцин'),

    # КР819_1 — Холецистит
    ('КР819_1', '1',   'холецистит'),
    ('КР819_1', '2.2', 'патогенез'),
    ('КР819_1', '2.3', 'холецистита'),
    ('КР819_1', '3.1', 'холецистит'),

    # КР669_2 — Железодефицитная анемия
    ('КР669_2', '3.1.1', 'железа'),
    ('КР669_2', '3.1.3', 'трансфузи'),

    # КР1027_1 — АИГА
    ('КР1027_1', '1.1', 'аутоиммунная'),
    ('КР1027_1', '1.2', 'этиологи'),
    ('КР1027_1', '3.2', 'ритуксимаб'),
]


def check_spot(content: list, section_num: str, phrase: str) -> Optional[str]:
    """Return error string if phrase not found, else None."""
    smap = section_map(content)
    # Collect text from target section + all children
    texts = []
    for s in content:
        n = s['number']
        if n == section_num or n.startswith(section_num + '.'):
            texts.append((s.get('content') or '').lower())

    if not texts:
        return f"Section '{section_num}' not found in content"

    combined = ' '.join(texts)
    if phrase.lower() not in combined:
        return f"Phrase {phrase!r} not found in section '{section_num}'"
    return None


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_tests(verbose: bool = False) -> None:
    test_dir = Path('docs/test_parsed')
    stems = sorted({
        p.stem.replace('_content', '')
        for p in test_dir.glob('*.json')
        if not p.stem.endswith('_content')
    })

    total = 0
    failures = 0

    def report(name: str, check: str, errors: List[str]) -> None:
        nonlocal total, failures
        for e in errors:
            total += 1
            failures += 1
            print(f'  FAIL  [{name}] {check}: {e}')
        if not errors and verbose:
            total += 1
            print(f'  pass  [{name}] {check}')
        if not errors:
            total += 1

    for stem in stems:
        headings = load(stem, '')
        content  = load(stem, '_content')

        if headings is None or content is None:
            print(f'  SKIP  [{stem}] — missing JSON files')
            continue

        if verbose:
            print(f'\n── {stem} ──')

        report(stem, 'no_duplicates',       check_no_duplicates(content))
        report(stem, 'toc_order',           check_order(headings, content))
        report(stem, 'no_leader_bleed',     check_no_leader_bleed(content))
        report(stem, 'content_not_none',    check_content_not_none(content))
        report(stem, 'numbers_match_toc',   check_numbers_match_toc(headings, content))

    # Spot checks
    if verbose:
        print('\n── Content spot-checks ──')
    for stem, sec, phrase in SPOT_CHECKS:
        content = load(stem, '_content')
        if content is None:
            print(f'  SKIP  [{stem}] — missing content file')
            continue
        err = check_spot(content, sec, phrase)
        errors = [err] if err else []
        report(stem, f'spot:{sec}:{phrase}', errors)

    print()
    print('─' * 50)
    if failures == 0:
        print(f'✓  All {total} checks passed.')
    else:
        print(f'✗  {failures}/{total} checks FAILED.')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    run_tests(verbose=verbose)
