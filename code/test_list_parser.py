"""
List Parser Tests
=================
Two layers of testing:

1. Unit tests on synthetic content strings (exact, deterministic).
2. Spot-checks on real parsed files (known lists that must exist).

Usage
-----
  python test_list_parser.py
  python test_list_parser.py --verbose
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from list_parser import extract_lists

# ─── Helpers ──────────────────────────────────────────────────────────────────

total = 0
failures = 0
verbose = '--verbose' in sys.argv or '-v' in sys.argv


def ok(name: str) -> None:
    global total
    total += 1
    if verbose:
        print(f'  pass  {name}')


def fail(name: str, reason: str) -> None:
    global total, failures
    total += 1
    failures += 1
    print(f'  FAIL  {name}: {reason}')


def assert_true(cond: bool, name: str, reason: str = '') -> None:
    if cond:
        ok(name)
    else:
        fail(name, reason or 'condition is False')


def assert_eq(actual, expected, name: str) -> None:
    if actual == expected:
        ok(name)
    else:
        fail(name, f'expected {expected!r}, got {actual!r}')


def assert_items_contain(items: List[str], phrase: str, name: str) -> None:
    """At least one item must contain phrase (case-insensitive)."""
    if any(phrase.lower() in it.lower() for it in items):
        ok(name)
    else:
        fail(name, f'no item containing {phrase!r} in {items[:3]}…')


# ─── Unit tests ───────────────────────────────────────────────────────────────

def test_unicode_bullets() -> None:
    """Basic unicode bullet list."""
    content = (
        "Основные причины: "
        "• дефицит железа при рождении; "
        "• алиментарный дефицит железа; "
        "• повышенные потребности организма в железе."
    )
    lists = extract_lists(content)
    assert_true(len(lists) == 1, 'unicode_bullets/count', f'expected 1 list, got {len(lists)}')
    if lists:
        lst = lists[0]
        assert_eq(lst['type'], 'bullet', 'unicode_bullets/type')
        assert_eq(len(lst['items']), 3, 'unicode_bullets/item_count')
        assert_items_contain(lst['items'], 'дефицит железа при рождении', 'unicode_bullets/item0')
        assert_items_contain(lst['items'], 'алиментарный', 'unicode_bullets/item1')
        assert_items_contain(lst['items'], 'повышенные потребности', 'unicode_bullets/item2')


def test_dash_list_with_semicolons() -> None:
    """Dash list where items end with semicolons — should be detected."""
    content = (
        "Формы заболевания: "
        "- экзема истинная; "
        "- экзема микробная; "
        "- экзема себорейная; "
        "- экзема детская."
    )
    lists = extract_lists(content)
    assert_true(len(lists) >= 1, 'dash_semicolons/found', f'expected ≥1 list, got {len(lists)}')
    if lists:
        assert_items_contain(lists[0]['items'], 'экзема', 'dash_semicolons/items_have_экзема')
        assert_true(len(lists[0]['items']) >= 3, 'dash_semicolons/item_count')


def test_dash_in_definition_not_a_list() -> None:
    """Em-dash used in running text definitions must NOT produce a list."""
    content = (
        "Холецистит – воспалительное поражение ЖП. "
        "Острый холецистит – острое воспаление ЖП. "
        "Хронический холецистит – хроническое воспаление ЖП."
    )
    lists = extract_lists(content)
    assert_true(len(lists) == 0, 'dash_definition/no_list',
                f'should produce no lists, got {len(lists)}: {lists}')


def test_numbered_list() -> None:
    """Numbered list with ) marker."""
    content = (
        "Диагноз устанавливается на основании: "
        "1) анализа жалоб; "
        "2) анамнестических данных; "
        "3) физикального обследования; "
        "4) результатов биопсийного исследования."
    )
    lists = extract_lists(content)
    assert_true(len(lists) >= 1, 'numbered/found', f'expected ≥1 list, got {len(lists)}')
    if lists:
        num_lists = [l for l in lists if l['type'] == 'numbered']
        assert_true(len(num_lists) >= 1, 'numbered/has_numbered_type')
        if num_lists:
            assert_true(len(num_lists[0]['items']) >= 3, 'numbered/item_count')
            assert_items_contain(num_lists[0]['items'], 'анализ', 'numbered/item0')


def test_no_list_in_plain_text() -> None:
    """Plain text without any list markers must produce empty lists."""
    content = (
        "ОРВИ – самая частая инфекция человека. "
        "Дети в возрасте до 5 лет переносят в среднем 6-8 эпизодов ОРВИ в год. "
        "Заболеваемость наиболее высока в период с сентября по апрель."
    )
    lists = extract_lists(content)
    assert_eq(lists, [], 'plain_text/no_list')


def test_bibliography_suppressed() -> None:
    """Numbered items matching citation pattern must not produce a numbered list."""
    content = (
        "1. Иванов А.А. // Вестник медицины. 2020; Vol. 5: 10-15. "
        "2. Smith B. et al. doi:10.1234/abc. 2019. "
        "3. Петров В.В. // Журнал. 2021. pp. 20-25."
    )
    lists = extract_lists(content)
    num_lists = [l for l in lists if l['type'] == 'numbered']
    assert_eq(num_lists, [], 'bibliography/suppressed')


def test_empty_content() -> None:
    """Empty/None content must return empty list without error."""
    assert_eq(extract_lists(''), [], 'empty_string')
    assert_eq(extract_lists('   '), [], 'whitespace_only')


def test_minimum_items_threshold() -> None:
    """A single bullet item must NOT produce a list (below MIN_LIST_SIZE=2)."""
    content = "Рекомендуется: • только один пункт в этом тексте и больше ничего."
    lists = extract_lists(content)
    assert_eq(lists, [], 'single_item/no_list')


def test_mixed_lists() -> None:
    """Content with both bullet and numbered lists should produce two list objects."""
    content = (
        "Клинические признаки: "
        "• анемический синдром; "
        "• желтуха и бледность; "
        "• темная моча. "
        "Диагностика проводится: "
        "1) клинически; "
        "2) лабораторно; "
        "3) инструментально."
    )
    lists = extract_lists(content)
    types = {l['type'] for l in lists}
    assert_true('bullet' in types, 'mixed/has_bullet')
    assert_true('numbered' in types, 'mixed/has_numbered')


# ─── Spot-checks on real files ────────────────────────────────────────────────

SPOT_CHECKS = [
    # (file_stem, section_number, expected_type, min_items, phrase_in_any_item)
    ('КР669_2', '1.2', 'bullet', 4, 'железа'),
    # КР669_2/1.5 is a classification table flattened to inline text — not detectable as a list
    ('КР669_2', '2.2', 'bullet', 5, 'изменения'),
    ('КР246_3', '1.5', 'bullet', 3, 'экзема'),
    ('КР246_3', '2',   'numbered', 3, 'анализа'),
    ('КР819_1', '2.2', 'numbered', 2, None),
    ('КР1027_1', '1.6', 'bullet', 3, 'синдром'),
    ('КР1027_1', '2.3', 'bullet', 5, 'Рекомендуется'),
]


def run_spot_checks() -> None:
    for stem, sec_num, exp_type, min_items, phrase in SPOT_CHECKS:
        path = Path(f'docs/test_parsed/{stem}_content.json')
        if not path.exists():
            fail(f'spot:{stem}/{sec_num}', 'content file not found')
            continue

        sections = json.loads(path.read_text(encoding='utf-8'))
        sec = next((s for s in sections if s['number'] == sec_num), None)
        if sec is None:
            fail(f'spot:{stem}/{sec_num}', 'section not found')
            continue

        lists = sec.get('lists', [])
        matched = [l for l in lists if l['type'] == exp_type]

        name = f'spot:{stem}/{sec_num}/{exp_type}'
        assert_true(len(matched) >= 1, name + '/found',
                    f'no {exp_type} list in section (lists={[l["type"] for l in lists]})')

        if matched:
            items = matched[0]['items']
            assert_true(len(items) >= min_items, name + f'/min_items>={min_items}',
                        f'only {len(items)} items, expected ≥{min_items}')
            if phrase:
                assert_items_contain(items, phrase, name + f'/phrase:{phrase}')


# ─── Runner ───────────────────────────────────────────────────────────────────

def main() -> None:
    if verbose:
        print('\n── Unit tests ──')

    test_unicode_bullets()
    test_dash_list_with_semicolons()
    test_dash_in_definition_not_a_list()
    test_numbered_list()
    test_no_list_in_plain_text()
    test_bibliography_suppressed()
    test_empty_content()
    test_minimum_items_threshold()
    test_mixed_lists()

    if verbose:
        print('\n── Spot-checks ──')

    run_spot_checks()

    print()
    print('─' * 50)
    if failures == 0:
        print(f'✓  All {total} checks passed.')
    else:
        print(f'✗  {failures}/{total} checks FAILED.')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
