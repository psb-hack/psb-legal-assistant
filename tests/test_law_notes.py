from datetime import date

import pytest

from lex_documents import AmendmentNote, LawAddress, LawUnit, LawUnitKind
from lex_law_notes import LawSource, read_amendment, read_source

ACT = "Федеральный закон от 2 марта 2031 г. № 77-ФЗ"


def _address(kind: LawUnitKind, number: str | None) -> LawAddress:
    return LawAddress((LawUnit(kind, number),), None)


def test_code_is_named_by_its_abbreviation() -> None:
    source = read_source(
        [
            "Источник: Отпуск. Статья 300. ТК РФ Учебный отпуск. Трудовой кодекс "
            "Российской Федерации от 30.12.2001 N 197-ФЗ (ТК РФ) "
            "(с изменениями и дополнениями)",
            "URL: https://example.org/labour/300",
            "Дата выгрузки: 05.06.2031",
        ]
    )
    assert source == LawSource(
        act="Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ (ТК РФ)",
        act_short="ТК РФ",
        url="https://example.org/labour/300",
        retrieved=date(2031, 6, 5),
    )


def test_federal_law_is_named_by_its_number() -> None:
    source = read_source(
        [
            'Источник: Статья 2. Федеральный закон от 01.02.2030 N 500-ФЗ "О базовой '
            'ставке" (с изменениями и дополнениями)'
        ]
    )
    assert source.act == 'Федеральный закон от 01.02.2030 N 500-ФЗ "О базовой ставке"'
    assert source.act_short == "500-ФЗ"
    assert source.url == ""
    assert source.retrieved is None


def test_constitutional_law_is_named_by_its_number() -> None:
    source = read_source(
        ["Источник: Статья 4. Федеральный конституционный закон от 01.02.2030 № 3-ФКЗ"]
    )
    assert source.act_short == "3-ФКЗ"


def test_act_without_abbreviation_and_number_keeps_its_name() -> None:
    assert read_source(["Источник: Статья 1. Устав города"]).act_short == "Устав города"


@pytest.mark.parametrize(
    ("line", "target"),
    [
        (
            f"Пункт 3 изменен с 1 июля 2031 г. - {ACT}",
            _address(LawUnitKind.CLAUSE, "3"),
        ),
        (
            f"Пункт 1.8-3 изменен с 1 июля 2031 г. - {ACT}",
            _address(LawUnitKind.CLAUSE, "1.8-3"),
        ),
        (
            f"Подпункт 2.1 изменен с 1 июля 2031 г. - {ACT}",
            _address(LawUnitKind.SUBCLAUSE, "2.1"),
        ),
        (f"Часть 1 изменена с 1 июля 2031 г. - {ACT}", _address(LawUnitKind.PART, "1")),
        (
            f"Пункт 1 дополнен подпунктом 1.2 с 1 июля 2031 г. - {ACT}",
            _address(LawUnitKind.SUBCLAUSE, "1.2"),
        ),
        (
            f"Статья 13.11 дополнена частью 10 с 1 июля 2031 г. - {ACT}",
            _address(LawUnitKind.PART, "10"),
        ),
        (
            f"Статья 284 дополнена пунктом 1.8-3 с 1 июля 2031 г. - {ACT}",
            _address(LawUnitKind.CLAUSE, "1.8-3"),
        ),
        (
            f"Примечания изменены с 1 июля 2031 г. - {ACT}",
            _address(LawUnitKind.NOTE, None),
        ),
        (f"Статья 59 изменена с 1 июля 2031 г. - {ACT}", None),
    ],
)
def test_amendment_names_its_unit(line: str, target: LawAddress | None) -> None:
    assert read_amendment(line) == AmendmentNote(
        target=target, effective_from=date(2031, 7, 1), source_act=ACT, text=line
    )


def test_amendment_is_read_with_latin_number_sign_and_non_breaking_spaces() -> None:
    line = (
        "Пункт 3 изменен с 1 июля 2031 г. - "
        "Федеральный закон от 2 марта 2031 г. N 77-ФЗ"
    )
    note = read_amendment(line)
    assert note is not None
    assert note.effective_from == date(2031, 7, 1)
    assert note.source_act == "Федеральный закон от 2 марта 2031 г. N 77-ФЗ"


def test_amendment_without_source_act_is_still_an_amendment() -> None:
    note = read_amendment("Пункт 3 изменен с 1 июля 2031 г.")
    assert note is not None
    assert note.source_act is None


@pytest.mark.parametrize(
    "comment",
    [
        "Федеральным законом от 1 марта 2030 г. N 11-ФЗ в статью 5 внесены изменения, "
        "вступающие в силу с 1 января 2031 г.",
        "О конституционно-правовом смысле пункта 3 см. постановление от 1 июня 2030 г. N 5-П",
        "Положения пункта 3 (в редакции закона от 1 мая 2030 г. N 7-ФЗ) применяются "
        "с 1 июля 2031 г.",
        "Пункт 3 изменен с 31 февраля 2031 г. - Федеральный закон",
    ],
)
def test_comment_without_address_is_not_an_amendment(comment: str) -> None:
    assert read_amendment(comment) is None
