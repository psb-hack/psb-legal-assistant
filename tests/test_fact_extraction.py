from datetime import date
from fractions import Fraction

import pytest

from lex_dates import PartialDate
from lex_fact_extraction import (
    Bound,
    DateFact,
    Quantity,
    QuantityKind,
    Unit,
    Vat,
    find_facts,
    same_fact,
)


def _quantities(text: str) -> list[Quantity]:
    return [fact for fact in find_facts(text) if isinstance(fact, Quantity)]


def _one(text: str) -> Quantity:
    (quantity,) = _quantities(text)
    return quantity


def _read(text: str) -> list[tuple[Fraction, Unit, Bound]]:
    return [(one.value, one.unit, one.bound) for one in _quantities(text)]


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("450 000 рублей", 450_000, Unit.RUBLE),
        ("450 000 руб.", 450_000, Unit.RUBLE),
        ("450 тыс. руб.", 450_000, Unit.RUBLE),
        ("0,45 млн рублей", 450_000, Unit.RUBLE),
        ("четыреста пятьдесят тысяч рублей", 450_000, Unit.RUBLE),
        ("10 000 000 (десяти миллионов) рублей", 10_000_000, Unit.RUBLE),
        ("12 миллионов рублей", 12_000_000, Unit.RUBLE),
        ("1,5 млрд ₽", 1_500_000_000, Unit.RUBLE),
        ("0,1%", Fraction(1, 10), Unit.PERCENT),
        ("0,1 процента", Fraction(1, 10), Unit.PERCENT),
        ("0.1 %", Fraction(1, 10), Unit.PERCENT),
        ("7% (семь процентов)", 7, Unit.PERCENT),
        ("30 (тридцати) календарных дней", 30, Unit.CALENDAR_DAY),
        ("5 рабочих дней", 5, Unit.WORKING_DAY),
        ("трёх банковских дней", 3, Unit.BANKING_DAY),
        ("5 дней", 5, Unit.DAY),
        ("сорока пяти дней", 45, Unit.DAY),
        ("2 недели", 2, Unit.WEEK),
        ("шести месяцев", 6, Unit.MONTH),
        ("5 (пять) лет", 5, Unit.YEAR),
        ("одного года", 1, Unit.YEAR),
        ("24 часа", 24, Unit.HOUR),
        ("полутора месяцев", Fraction(3, 2), Unit.MONTH),
        ("500 000", 500_000, Unit.NONE),
        ("в 3 экземплярах", 3, Unit.NONE),
    ],
)
def test_quantity_is_read_with_value_and_unit(
    text: str, value: Fraction, unit: Unit
) -> None:
    assert _read(text) == [(value, unit, Bound.EXACT)]


def test_digits_with_spelled_duplicate_are_one_fact() -> None:
    text = "450 000 (четыреста пятьдесят тысяч) рублей в месяц"
    quantity = _one(text)
    assert quantity.value == 450_000
    assert text[quantity.start : quantity.end] == (
        "450 000 (четыреста пятьдесят тысяч) рублей"
    )


def test_brackets_without_numeral_stay_outside_quantity() -> None:
    text = "450 000 (включая НДС) рублей"
    assert text[_one(text).start : _one(text).end] == "450 000"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("450 000 рублей", QuantityKind.MONEY),
        ("7%", QuantityKind.PERCENT),
        ("5 рабочих дней", QuantityKind.DURATION),
        ("3 года", QuantityKind.DURATION),
        ("5-го числа", QuantityKind.DAY_OF_MONTH),
        ("500 000", QuantityKind.NUMBER),
    ],
)
def test_quantity_kind_follows_unit(text: str, kind: QuantityKind) -> None:
    assert _one(text).kind is kind


@pytest.mark.parametrize(
    ("text", "bound"),
    [
        ("не более чем на 7%", Bound.AT_MOST),
        ("на срок не более 3 (трёх) рабочих дней", Bound.AT_MOST),
        ("не может превышать шести месяцев", Bound.AT_MOST),
        ("не превышает 10 000 рублей", Bound.AT_MOST),
        ("максимум 5 дней", Bound.AT_MOST),
        ("не менее 30 календарных дней", Bound.AT_LEAST),
        ("свыше 10 000 000 рублей", Bound.ABOVE),
        ("более 5 лет", Bound.ABOVE),
        ("превышает 15 млн рублей", Bound.ABOVE),
        ("менее 100 000 рублей", Bound.BELOW),
        ("до 7%", Bound.UP_TO),
        ("порог — 15 000 000 рублей", Bound.THRESHOLD),
        ("порог: 10 млн рублей", Bound.THRESHOLD),
        ("лимит в 5 млн рублей", Bound.THRESHOLD),
        ("в течение 5 рабочих дней", Bound.EXACT),
        ("7%", Bound.EXACT),
    ],
)
def test_operator_is_part_of_quantity(text: str, bound: Bound) -> None:
    assert _one(text).bound is bound


def test_operator_belongs_to_nearest_number_only() -> None:
    assert _read("не более 5 дней, затем 10 дней") == [
        (5, Unit.DAY, Bound.AT_MOST),
        (10, Unit.DAY, Bound.EXACT),
    ]


@pytest.mark.parametrize(
    "text", ["от 3 до 5 рабочих дней", "3–5 рабочих дней", "3 - 5 рабочих дней"]
)
def test_range_is_one_quantity_with_two_bounds(text: str) -> None:
    quantity = _one(text)
    assert (quantity.value, quantity.upper) == (3, 5)
    assert (quantity.unit, quantity.bound) == (Unit.WORKING_DAY, Bound.RANGE)


def test_change_from_one_value_to_another_is_not_range() -> None:
    assert _read("увеличить с 450 000 до 480 000 рублей") == [
        (450_000, Unit.NONE, Bound.EXACT),
        (480_000, Unit.RUBLE, Bound.UP_TO),
    ]


@pytest.mark.parametrize(
    ("text", "vat"),
    [
        ("450 000 рублей в месяц, включая НДС", Vat.INCLUDED),
        ("450 000 рублей, в том числе НДС 20%", Vat.INCLUDED),
        ("450 000 рублей с НДС", Vat.INCLUDED),
        ("450 000 рублей без НДС", Vat.EXCLUDED),
        ("450 000 рублей, кроме того НДС", Vat.EXCLUDED),
        ("450 000 рублей", None),
        ("450 000 рублей. Цена включая НДС указана отдельно", None),
    ],
)
def test_vat_mark_is_kept_with_money(text: str, vat: Vat | None) -> None:
    assert _quantities(text)[0].vat is vat


@pytest.mark.parametrize(
    ("text", "value", "bound"),
    [
        ("не позднее 5-го числа текущего месяца", 5, Bound.AT_MOST),
        ("до 25 числа", 25, Bound.UP_TO),
        ("не позднее пятого числа", 5, Bound.AT_MOST),
        ("двадцать первого числа", 21, Bound.EXACT),
    ],
)
def test_day_of_month_is_separate_unit(text: str, value: int, bound: Bound) -> None:
    assert _read(text) == [(value, Unit.DAY_OF_MONTH, bound)]


@pytest.mark.parametrize(
    "text",
    [
        "п. 3.1 договора",
        "пп. 3.1, 3.2 и 3.4",
        "в соответствии с пунктом 3.4",
        "ст. 13.11 КоАП",
        "статьи 70 ТК РФ",
        "ч. 5 ст. 70",
        "договор № П-17/2024",
        "договор П-17/2024",
        "закон 152-ФЗ",
        "положение 14-П",
        "склад № 3",
        "каб. 412",
        "офис 12",
        "ИНН 7701234567",
        "ОГРН 1027700132195",
        "раздел 3.1",
        "перечень: 1) счёт; 2) акт",
        "30-дневный срок",
    ],
)
def test_requisites_and_addresses_are_not_quantities(text: str) -> None:
    assert find_facts(text) == ()


def test_number_in_closing_bracket_is_not_list_marker() -> None:
    assert _read("пени (но не более 10) начисляются") == [
        (10, Unit.NONE, Bound.AT_MOST)
    ]


def test_quantity_next_to_requisite_is_still_found() -> None:
    assert _read("по п. 3.1 договора № А-08/2024 плата — 450 000 рублей") == [
        (450_000, Unit.RUBLE, Bound.EXACT)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "один из участников",
        "не чаще одного раза в год",
        "две стороны договора",
        "трёх лиц",
        "стороны одновременно подписали акт",
    ],
)
def test_small_spelled_number_without_unit_is_not_fact(text: str) -> None:
    assert find_facts(text) == ()


def test_spelled_number_from_ten_is_fact_without_unit() -> None:
    assert _read("в четырнадцати случаях") == [(14, Unit.NONE, Bound.EXACT)]


@pytest.mark.parametrize(
    ("text", "period", "bound"),
    [
        ("1 августа 2024 г.", PartialDate.of_day(date(2024, 8, 1)), Bound.EXACT),
        ("01.08.2024", PartialDate.of_day(date(2024, 8, 1)), Bound.EXACT),
        ("в августе 2024 года", PartialDate.of_month(2024, 8), Bound.EXACT),
        ("в 2024 году", PartialDate.of_year(2024), Bound.EXACT),
        (
            "первого августа 2024 года",
            PartialDate.of_day(date(2024, 8, 1)),
            Bound.EXACT,
        ),
        (
            "не позднее двадцать первого марта 2025 года",
            PartialDate.of_day(date(2025, 3, 21)),
            Bound.AT_MOST,
        ),
        ("до 31.12.2025", PartialDate.of_day(date(2025, 12, 31)), Bound.UP_TO),
    ],
)
def test_date_is_fact_with_precision(
    text: str, period: PartialDate, bound: Bound
) -> None:
    (fact,) = find_facts(text)
    assert isinstance(fact, DateFact)
    assert (fact.period, fact.bound) == (period, bound)
    assert (fact.start, fact.end) != (0, 0)


def test_year_of_date_is_not_read_as_number() -> None:
    facts = find_facts("с 1 марта 2025 года плата — 480 000 рублей")
    assert [type(fact) for fact in facts] == [DateFact, Quantity]


def test_facts_come_in_text_order_with_positions() -> None:
    text = "Плата 450 000 рублей вносится не позднее 5-го числа с 01.03.2025."
    facts = find_facts(text)
    assert [text[fact.start : fact.end] for fact in facts] == [
        "450 000 рублей",
        "5-го числа",
        "01.03.2025",
    ]


def _same(first: str, second: str) -> bool:
    (one,) = find_facts(first)
    (other,) = find_facts(second)
    return same_fact(one, other)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("450 000 рублей", "450 тыс. руб."),
        ("450 000 рублей", "четыреста пятьдесят тысяч рублей"),
        ("450 000 (четыреста пятьдесят тысяч) рублей", "0,45 млн рублей"),
        ("0,1%", "0,1 процента"),
        ("не более 30 календарных дней", "не позднее тридцати календарных дней"),
        ("не может превышать шести месяцев", "не более 6 месяцев"),
        ("1 августа 2024 года", "01.08.2024"),
        ("5-го числа", "пятого числа"),
    ],
)
def test_same_fact_in_different_wording(first: str, second: str) -> None:
    assert _same(first, second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("5 рабочих дней", "5 дней"),
        ("5 дней", "5 календарных дней"),
        ("5 рабочих дней", "5 банковских дней"),
        ("5 рабочих дней", "5 календарных дней"),
        ("до 7%", "7%"),
        ("не более 7%", "не менее 7%"),
        ("450 000 рублей", "450 000"),
        ("450 000 рублей", "480 000 рублей"),
        ("450 000 рублей с НДС", "450 000 рублей без НДС"),
        ("24 месяца", "2 года"),
        ("в августе 2024 года", "1 августа 2024 года"),
        ("5-го числа", "5 дней"),
        ("в 2024 году", "2024 рубля"),
    ],
)
def test_different_facts_are_not_same(first: str, second: str) -> None:
    assert not _same(first, second)
