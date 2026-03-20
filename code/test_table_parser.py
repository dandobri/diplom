"""
test_table_parser.py
====================
Tests for table_parser.py.

Unit tests
----------
- Filter functions (_is_quality_checklist, _is_evidence_legend,
  _is_layout_artifact, _is_body_text_wrap)

Spot-checks on pre-parsed _content.json files
---------------------------------------------
- КР819_1 sec 3  → 1-col diagnostic-criteria table present
- КР819_1 sec 8  → multi-col symptom-informativeness table present
- КР669_2 sec 1.6 → iron-deficiency staging table present
- КР1027_1 sec 1.2 → AIHA type classification table present
- КР621_3 sec 1.2 → etiology numbered table present
- Quality-checklist tables should NOT appear anywhere

Usage
-----
  python test_table_parser.py [--verbose]
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from table_parser import (
    _is_quality_checklist,
    _is_evidence_legend,
    _is_layout_artifact,
    _is_body_text_wrap,
    _is_diagnostic_header,
    _should_skip,
)

# ── Result tracking ───────────────────────────────────────────────────────────

VERBOSE = '--verbose' in sys.argv
_passed = 0
_failed = 0


def ok(name: str) -> None:
    global _passed
    _passed += 1
    if VERBOSE:
        print(f"  pass  {name}")


def fail(name: str, reason: str = '') -> None:
    global _failed
    _failed += 1
    msg = f"  FAIL  {name}"
    if reason:
        msg += f": {reason}"
    print(msg)


def assert_true(cond: bool, name: str, reason: str = '') -> None:
    if cond:
        ok(name)
    else:
        fail(name, reason)


def assert_false(cond: bool, name: str, reason: str = '') -> None:
    assert_true(not cond, name, reason)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_quality_checklist_standard() -> None:
    """Standard да/нет quality checklist."""
    t = [
        ['№ п/п', 'Критерии качества', 'Оценка выполнения'],
        ['1.', 'Выполнен общий анализ крови', 'Да/Нет'],
        ['2.', 'Проведено ЭКГ', 'Да/Нет'],
        ['3.', 'Назначены препараты', 'Да/Нет'],
    ]
    assert_true(_is_quality_checklist(t), 'checklist/standard/да_нет')
    assert_true(_should_skip(t), 'checklist/standard/should_skip')


def test_quality_checklist_uur_variant() -> None:
    """Quality checklist using УУР/УДД columns instead of да/нет."""
    t = [
        ['№', 'Критерии качества', 'Уровень\nубедитель-\nности реко-\nмендаций', 'Уровень\nдостовер-\nности дока-\nзательств'],
        ['1.', 'Выполнен ОАК...', 'С', '4'],
        ['2.', 'Выполнено УЗИ...', 'С', '5'],
        ['3.', 'Назначена терапия...', 'В', '3'],
    ]
    assert_true(_is_quality_checklist(t), 'checklist/uur_variant')
    assert_true(_should_skip(t), 'checklist/uur_variant/should_skip')


def test_evidence_legend_standard() -> None:
    """УУР/УДД decoding legend."""
    t = [
        ['УУР', 'Расшифровка'],
        ['A', 'Сильная рекомендация (все рассматриваемые критерии эффективности)'],
        ['B', 'Условная рекомендация'],
        ['C', 'Слабая рекомендация'],
    ]
    assert_true(_is_evidence_legend(t), 'evidence/uur_legend')
    assert_true(_should_skip(t), 'evidence/uur_legend/should_skip')


def test_evidence_legend_udd() -> None:
    """УДД legend starting with УДД."""
    t = [
        ['УДД', 'Расшифровка'],
        ['1', 'Систематические обзоры исследований с контролем'],
        ['2', 'Отдельные исследования с контролем'],
        ['3', 'Исследования без последовательного контроля'],
    ]
    assert_true(_is_evidence_legend(t), 'evidence/udd_legend')


def test_layout_artifact_single_row() -> None:
    t = [['Холецистит – воспалительное поражение ЖП.']]
    assert_true(_is_layout_artifact(t), 'layout/single_row')
    assert_true(_should_skip(t), 'layout/single_row/should_skip')


def test_layout_artifact_two_rows() -> None:
    t = [['Row 1', 'value'], ['Row 2', 'value']]
    assert_true(_is_layout_artifact(t), 'layout/two_rows')


def test_layout_artifact_not_triggered() -> None:
    """3+ row table with content should not be flagged as artifact."""
    t = [
        ['Симптом', 'Информативность'],
        ['Симптом Мерфи', '65%'],
        ['Лихорадка', '35%'],
        ['Боль', '81%'],
    ]
    assert_false(_is_layout_artifact(t), 'layout/ok_table_not_artifact')


def test_body_text_wrap_first_col_dominant() -> None:
    """2-col table where first column has all the content."""
    t = [
        ['Пациентам с клиническими проявлениями рекомендуется', ''],
        ['выполнить общий анализ крови [27].', ''],
        ['Уровень убедительности – В (уровень достоверности – 1)', ''],
        ['Комментарий. Дополнительная информация...', ''],
    ]
    assert_true(_is_body_text_wrap(t), 'body_wrap/first_col_dominant')
    assert_true(_should_skip(t), 'body_wrap/first_col_dominant/should_skip')


def test_body_text_wrap_second_col_dominant() -> None:
    """2-col table where second column has all the content."""
    t = [
        ['', '1. Все пациенты должны быть осмотрены'],
        ['', '2. Выявление признаков служит показанием'],
        ['', '3. При неэффективности лечения...'],
        ['', '4. Следует избегать'],
    ]
    assert_true(_is_body_text_wrap(t), 'body_wrap/second_col_dominant')


def test_body_text_wrap_not_triggered_on_real_table() -> None:
    """Balanced multi-col table should not be flagged."""
    t = [
        ['Симптомы', 'Чувствительность, %', 'Специфичность, %'],
        ['Симптом Мерфи', '65', '87'],
        ['Лихорадка', '35', '80'],
        ['Боль в правом квадранте', '81', '67'],
    ]
    assert_false(_is_body_text_wrap(t), 'body_wrap/real_table_not_wrapped')


def test_diagnostic_header_criteria() -> None:
    assert_true(_is_diagnostic_header('Критерии установления диагноза «Острый холецистит»'),
                'diag_header/criteria')
    assert_true(_is_diagnostic_header('Дифференциальный диагноз при подозрении'),
                'diag_header/diff_diagnosis')
    assert_true(_is_diagnostic_header('Степень тяжести холецистита'),
                'diag_header/severity')


def test_diagnostic_header_negative() -> None:
    assert_false(_is_diagnostic_header('Кузьминова Жанна Андреевна — кандидат медицинских наук'),
                 'diag_header/author_name_not_diagnostic')
    assert_false(_is_diagnostic_header('Уровень убедительности рекомендаций – C'),
                 'diag_header/uur_not_diagnostic')


# ── Spot-checks on parsed JSON ────────────────────────────────────────────────

PARSED_DIR = Path('docs/test_parsed')


def _load_section(stem: str, sec_num: str):
    path = PARSED_DIR / f'{stem}_content.json'
    if not path.exists():
        return None
    sections = json.loads(path.read_text(encoding='utf-8'))
    return next((s for s in sections if s['number'] == sec_num), None)


def _has_table_with_phrase(tables: list, phrase: str) -> bool:
    """True if any table contains phrase in any cell."""
    phrase_lo = phrase.lower()
    for t in tables:
        for row in t.get('rows', []):
            for cell in row:
                if phrase_lo in (cell or '').lower():
                    return True
    return False


def _no_checklist_tables_in_sections(stem: str) -> bool:
    """Verify no quality-checklist tables leaked through."""
    path = PARSED_DIR / f'{stem}_content.json'
    if not path.exists():
        return True
    sections = json.loads(path.read_text(encoding='utf-8'))
    for s in sections:
        for t in s.get('tables', []):
            rows = t.get('rows', [])
            if _is_quality_checklist(rows):
                return False
    return True


def run_spot_checks() -> None:
    # КР819_1: 1-col diagnostic-criteria table in section 3
    name = 'spot:КР819_1/sec3/diag_criteria'
    sec = _load_section('КР819_1', '3')
    if sec is None:
        fail(name, 'section not found')
    else:
        tbls = sec.get('tables', [])
        assert_true(len(tbls) >= 1, name + '/found',
                    f'expected ≥1 table, got {len(tbls)}')
        assert_true(
            _has_table_with_phrase(tbls, 'критерии'),
            name + '/contains_criteria',
        )

    # КР819_1: symptom-informativeness appendix table in section 8
    name8 = 'spot:КР819_1/sec8/symptom_table'
    sec8 = _load_section('КР819_1', '8')
    if sec8 is None:
        fail(name8, 'section not found')
    else:
        tbls8 = sec8.get('tables', [])
        assert_true(len(tbls8) >= 1, name8 + '/found',
                    f'expected ≥1 table, got {len(tbls8)}')
        assert_true(
            _has_table_with_phrase(tbls8, 'симптом'),
            name8 + '/contains_симптом',
        )

    # КР669_2: iron-deficiency staging table in section 1.6
    name_fe = 'spot:КР669_2/sec1.6/iron_table'
    sec_fe = _load_section('КР669_2', '1.6')
    if sec_fe is None:
        fail(name_fe, 'section not found')
    else:
        tbls_fe = sec_fe.get('tables', [])
        assert_true(len(tbls_fe) >= 1, name_fe + '/found',
                    f'expected ≥1 table, got {len(tbls_fe)}')
        assert_true(
            _has_table_with_phrase(tbls_fe, 'железо') or
            _has_table_with_phrase(tbls_fe, 'ферритин') or
            _has_table_with_phrase(tbls_fe, 'дефицит'),
            name_fe + '/contains_iron_keyword',
        )

    # КР1027_1: AIHA classification table in section 1.2
    name_ai = 'spot:КР1027_1/sec1.2/aiha_table'
    sec_ai = _load_section('КР1027_1', '1.2')
    if sec_ai is None:
        fail(name_ai, 'section not found')
    else:
        tbls_ai = sec_ai.get('tables', [])
        assert_true(len(tbls_ai) >= 1, name_ai + '/found',
                    f'expected ≥1 table, got {len(tbls_ai)}')
        assert_true(
            _has_table_with_phrase(tbls_ai, 'аига') or
            _has_table_with_phrase(tbls_ai, 'аутоантитела') or
            _has_table_with_phrase(tbls_ai, 'тип'),
            name_ai + '/contains_aiha_keyword',
        )

    # КР621_3: etiology numbered table in section 1.2
    name_et = 'spot:КР621_3/sec1.2/etiology_table'
    sec_et = _load_section('КР621_3', '1.2')
    if sec_et is None:
        fail(name_et, 'section not found')
    else:
        tbls_et = sec_et.get('tables', [])
        assert_true(len(tbls_et) >= 1, name_et + '/found',
                    f'expected ≥1 table, got {len(tbls_et)}')

    # Negative: no quality-checklist tables should have leaked
    for stem in ['КР819_1', 'КР669_2', 'КР1027_1', 'КР246_3',
                 'КР25_2', 'КР621_3', 'КР536_3', 'КР540_3']:
        assert_true(
            _no_checklist_tables_in_sections(stem),
            f'no_checklist/{stem}',
            'quality-checklist table found in sections',
        )

    # Negative: no evidence-legend tables should have leaked
    for stem in ['КР819_1', 'КР669_2', 'КР1027_1', 'КР621_3']:
        path = PARSED_DIR / f'{stem}_content.json'
        if path.exists():
            sections = json.loads(path.read_text(encoding='utf-8'))
            leak = False
            for s in sections:
                for t in s.get('tables', []):
                    if _is_evidence_legend(t.get('rows', [])):
                        leak = True
            assert_false(leak, f'no_evidence_legend/{stem}',
                         'UUR/UDD legend table found in sections')


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    if VERBOSE:
        print()
        print('── Unit tests ──')

    test_quality_checklist_standard()
    test_quality_checklist_uur_variant()
    test_evidence_legend_standard()
    test_evidence_legend_udd()
    test_layout_artifact_single_row()
    test_layout_artifact_two_rows()
    test_layout_artifact_not_triggered()
    test_body_text_wrap_first_col_dominant()
    test_body_text_wrap_second_col_dominant()
    test_body_text_wrap_not_triggered_on_real_table()
    test_diagnostic_header_criteria()
    test_diagnostic_header_negative()

    if VERBOSE:
        print()
        print('── Spot-checks ──')

    run_spot_checks()

    total = _passed + _failed
    print()
    print('─' * 50)
    if _failed == 0:
        print(f'✓  All {total} checks passed.')
    else:
        print(f'✗  {_failed}/{total} checks FAILED.')
    sys.exit(0 if _failed == 0 else 1)


if __name__ == '__main__':
    main()
