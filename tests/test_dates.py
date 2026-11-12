import re
from datetime import date

import pytest

from lex_dates import (
    DATE_PATTERN,
    MONTH_WORD_PATTERN,
    DateMention,
    DatePrecision,
    PartialDate,
    calendar_period,
    find_dates,
    month_number,
    parse_date,
)

ANY_DATE = re.compile(DATE_PATTERN, re.IGNORECASE)


def test_numeric_date_is_parsed() -> None:
    assert parse_date("01.03.2025") == date(2025, 3, 1)


def test_verbal_date_is_parsed() -> None:
    assert parse_date("1 августа 2024") == date(2024, 8, 1)


def test_month_name_in_capitals_is_parsed() -> None:
    assert parse_date("12 МАЯ 2030") == date(2030, 5, 12)


@pytest.mark.parametrize("text", ["31.02.2025", "00.01.2025", "15.13.2025"])
def test_impossible_calendar_date_is_rejected(text: str) -> None:
    assert parse_date(text) is None


@pytest.mark.parametrize("text", ["3.1", "август 2024", "от 01.03.2025"])
def test_text_that_is_not_exactly_a_date_is_rejected(text: str) -> None:
    assert parse_date(text) is None


def test_pattern_finds_both_notations_in_a_sentence() -> None:
    sentence = "Договор от 12 февраля 2024 г. изменён 03.09.2024."
    assert ANY_DATE.findall(sentence) == ["12 февраля 2024", "03.09.2024"]


def test_pattern_does_not_cut_a_date_out_of_a_longer_number() -> None:
    assert ANY_DATE.findall("счёт 4070.28.10100000001, код 123.01.20255") == []


ANY_MONTH = re.compile(MONTH_WORD_PATTERN, re.IGNORECASE)


@pytest.mark.parametrize(
    ("period", "end"),
    [
        (PartialDate.of_day(date(2024, 2, 29)), date(2024, 3, 1)),
        (PartialDate.of_month(2024, 2), date(2024, 3, 1)),
        (PartialDate.of_month(2024, 12), date(2025, 1, 1)),
        (PartialDate.of_year(2024), date(2025, 1, 1)),
    ],
)
def test_period_ends_on_the_first_day_after_it(period: PartialDate, end: date) -> None:
    assert period.end == end


def test_period_is_shifted_by_calendar_units_of_its_precision() -> None:
    assert PartialDate.of_month(2025, 1).shifted(-1) == PartialDate.of_month(2024, 12)
    assert PartialDate.of_month(2024, 11).shifted(14) == PartialDate.of_month(2026, 1)
    assert PartialDate.of_year(2025).shifted(-2) == PartialDate.of_year(2023)
    assert PartialDate.of_day(date(2025, 3, 1)).shifted(-1) == PartialDate.of_day(
        date(2025, 2, 28)
    )


def test_period_keeps_its_precision_and_first_day() -> None:
    period = PartialDate.of_month(2024, 8)
    assert (period.start, period.precision) == (date(2024, 8, 1), DatePrecision.MONTH)


def test_dates_of_every_precision_are_found_with_their_place_in_the_text() -> None:
    text = "С 01.03.2025 по 5 мая 2025 г., в августе 2024 года и в 2023 году."
    assert find_dates(text) == (
        DateMention(PartialDate.of_day(date(2025, 3, 1)), 2, 12),
        DateMention(PartialDate.of_day(date(2025, 5, 5)), 16, 26),
        DateMention(PartialDate.of_month(2024, 8), 33, 45),
        DateMention(PartialDate.of_year(2023), 55, 64),
    )


@pytest.mark.parametrize(
    ("text", "period"),
    [
        ("август 2024", PartialDate.of_month(2024, 8)),
        ("АВГУСТА 2024", PartialDate.of_month(2024, 8)),
        ("в мае\u00a02025", PartialDate.of_month(2025, 5)),
        ("за 2024 г.", PartialDate.of_year(2024)),
        ("с 2024 года", PartialDate.of_year(2024)),
    ],
)
def test_month_and_year_are_found_in_any_case(text: str, period: PartialDate) -> None:
    assert [mention.period for mention in find_dates(text)] == [period]


@pytest.mark.parametrize(
    "text",
    [
        "договор № П-17/2024",
        "штраф в 2000 рублей",
        "счёт 4070.28.10100000001",
        "31.02.2025 года",
        "31 февраля 2025 года",
        "2024 годовых отчёта",
        "мая",
    ],
)
def test_number_that_is_not_a_date_is_not_found(text: str) -> None:
    assert find_dates(text) == ()


def test_month_word_gives_its_number_in_three_cases() -> None:
    words = ANY_MONTH.findall("Январь, февраля, в марте, май, мая, в декабре")
    assert [month_number(word) for word in words] == [1, 2, 3, 5, 5, 12]


def test_month_word_is_not_cut_out_of_a_longer_word() -> None:
    assert ANY_MONTH.findall("мартовский отчёт, маяк, размах") == []


def test_calendar_period_is_a_day_or_a_whole_month() -> None:
    assert calendar_period(2024, 8, None) == PartialDate.of_month(2024, 8)
    assert calendar_period(2024, 8, "5") == PartialDate.of_day(date(2024, 8, 5))
    assert calendar_period(2025, 2, "31") is None
