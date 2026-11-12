from fractions import Fraction

import pytest

from lex_numerals import NumeralReading, multiplier_value, read_numeral, read_ordinal


def _value(text: str) -> Fraction | None:
    reading = read_numeral(text.split())
    return None if reading is None else reading.value


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("трёх", 3),
        ("тридцати", 30),
        ("сорока пяти", 45),
        ("четыреста пятьдесят тысяч", 450_000),
        ("одного миллиона", 1_000_000),
        ("десяти миллионов", 10_000_000),
        ("пятнадцати миллионов", 15_000_000),
        ("тысячи", 1000),
        ("двумястами тысячами", 200_000),
        ("ста двадцати", 120),
        ("девяноста", 90),
        ("одной тысячи пятисот", 1500),
        ("один миллион двести тысяч триста сорок пять", 1_200_345),
        ("двадцать одного", 21),
        ("шестью", 6),
        ("восьмьюдесятью", 80),
        ("Четырнадцати", 14),
        ("полтораста", 150),
    ],
)
def test_numeral_in_any_case_is_read(text: str, value: int) -> None:
    assert _value(text) == value


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("полутора", Fraction(3, 2)),
        ("полтора миллиона", Fraction(1_500_000)),
        ("ноль целых одна десятая", Fraction(1, 10)),
        ("две целых пять десятых", Fraction(5, 2)),
        ("ноль целых пять сотых", Fraction(1, 20)),
    ],
)
def test_fractional_numeral_is_read(text: str, value: Fraction) -> None:
    assert _value(text) == value


@pytest.mark.parametrize(
    ("text", "value", "words"),
    [
        # Внутри тройки разряды убывают, множители тоже: дальше — другое число.
        ("трёх пяти", 3, 1),
        ("сорок сорок", 40, 1),
        ("пятнадцать два", 15, 1),
        ("тысяча миллионов", 1000, 1),
        ("пяти рабочих дней", 5, 1),
        ("ноль целых рублей", 0, 1),
    ],
)
def test_numeral_ends_where_grammar_breaks(text: str, value: int, words: int) -> None:
    assert read_numeral(text.split()) == NumeralReading(Fraction(value), words)


@pytest.mark.parametrize(
    "text", ["стороны", "семьи", "одновременно", "трёхсторонней", "сотрудник", ""]
)
def test_word_that_only_resembles_numeral_is_not_number(text: str) -> None:
    assert read_numeral(text.split()) is None


@pytest.mark.parametrize(
    ("text", "value", "words"),
    [
        ("пятого числа", 5, 1),
        ("первое августа", 1, 1),
        ("третьего", 3, 1),
        ("восьмым", 8, 1),
        ("одиннадцатом", 11, 1),
        ("двадцать первого", 21, 2),
        ("тридцать первое", 31, 2),
    ],
)
def test_ordinal_day_is_read(text: str, value: int, words: int) -> None:
    assert read_ordinal(text.split()) == NumeralReading(Fraction(value), words)


@pytest.mark.parametrize("text", ["пять", "тридцать второго", "сорок первого", ""])
def test_cardinal_and_impossible_day_are_not_ordinals(text: str) -> None:
    assert read_ordinal(text.split()) is None


def test_multiplier_word_has_value() -> None:
    assert multiplier_value("миллионов") == 1_000_000
    assert multiplier_value("Тысяч") == 1000
    assert multiplier_value("рублей") is None
