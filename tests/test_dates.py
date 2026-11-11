import re
from datetime import date

import pytest

from lex_dates import DATE_PATTERN, parse_date

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
