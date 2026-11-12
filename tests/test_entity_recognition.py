"""Таблица распознавания на выдуманном каталоге: падежи, опечатки, пары-ловушки."""

from __future__ import annotations

import pytest

from lex_entity_catalog import Counterparty, build_catalog
from lex_entity_recognition import (
    Ambiguous,
    Mention,
    NotFound,
    Resolved,
    combine_turns,
    recognize,
)
from synthetic_counterparties import knowledge_base

CATALOG = build_catalog(knowledge_base())

PORT = "ООО «Дельта-Порт»"
PORTAL = "ООО «Дельта-Портал»"
PIER = "ООО «Южный Причал»"
OWN = "ООО «Горизонт»"
LOGISTIC = "ООО «Логистик»"
LOGISTICS = "ООО «Логистика»"
SUPPLY = "Договор поставки № Д-21/2031"
PORTAL_SUPPLY = "Договор поставки № ДП-04/2030"
CARRIAGE_2031 = "Договор перевозки № Л-01/2031"
CARRIAGE_2030 = "Договор перевозки № Л-01/2030"
MEMO = "Служебная записка № ОК-4 от 20.04.2031"

Summary = tuple[str, str, list[str], str | None]


def _summary(mention: Mention) -> Summary:
    resolution = mention.resolution
    if isinstance(resolution, Resolved):
        return (
            mention.text,
            "resolved",
            [_label(resolution.entity)],
            resolution.evidence.name,
        )
    if isinstance(resolution, Ambiguous):
        labels = [_label(candidate) for candidate in resolution.candidates]
        return mention.text, "ambiguous", labels, resolution.evidence.name
    return mention.text, "not_found", [_label(e) for e in resolution.similar], None


def _label(entity: object) -> str:
    return entity.name if isinstance(entity, Counterparty) else entity.title


def _recognized(text: str) -> list[Summary]:
    return [_summary(mention) for mention in recognize(text, CATALOG)]


@pytest.mark.parametrize(
    ("text", "surface", "name"),
    [
        ("Что с договором с ООО «Дельта-Порт»?", "Дельта-Порт", PORT),
        ("Претензия к «Дельта-Порту» готова", "Дельта-Порту", PORT),
        ("Мы недовольны Дельта-Портом", "Дельта-Портом", PORT),
        ("поставка от дельта-порта задержана", "дельта-порта", PORT),
        ("Дельта Порт прислал счёт", "Дельта Порт", PORT),
        ("ДЕЛЬТА-ПОРТ не отвечает", "ДЕЛЬТА-ПОРТ", PORT),
        ("С «Южным Причалом» расторгаем", "Южным Причалом", PIER),
        ("Аренда у южного причала дорогая", "южного причала", PIER),
        ("Оплата произведена Горизонтом", "Горизонтом", OWN),
        ("«Дельта-Портал» привёз технику", "Дельта-Портал", PORTAL),
        ("Счёт от Дельта-Портала", "Дельта-Портала", PORTAL),
        ("договор с Дельта-Порталом", "Дельта-Порталом", PORTAL),
        ("письмо в „Дельта-Портал“", "Дельта-Портал", PORTAL),
        ('письмо в "Дельта-Портал"', "Дельта-Портал", PORTAL),
    ],
)
def test_full_name_is_resolved_in_any_case_form(
    text: str, surface: str, name: str
) -> None:
    assert _recognized(text) == [(surface, "resolved", [name], "FULL_NAME")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Что у нас с «Дельтой»?",
            [("Дельтой", "ambiguous", [PORT, PORTAL], "PARTIAL")],
        ),
        (
            "Дельта задерживает поставку",
            [("Дельта", "ambiguous", [PORT, PORTAL], "PARTIAL")],
        ),
        ("«Портал» молчит", [("Портал", "resolved", [PORTAL], "PARTIAL")]),
        ("Что отгрузил Порт?", [("Порт", "resolved", [PORT], "PARTIAL")]),
        ("Звонили из Южного", [("Южного", "resolved", [PIER], "PARTIAL")]),
    ],
)
def test_part_of_a_name_names_every_counterparty_that_has_it(
    text: str, expected: list[Summary]
) -> None:
    assert _recognized(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Именительный падеж сильнее косвенного: родительный от «Логистик» не в счёт.
        ("Позвоните в «Логистика»", [("Логистика", "resolved", [LOGISTICS])]),
        ("договор с «Логистиком»", [("Логистиком", "resolved", [LOGISTIC])]),
        ("договор с «Логистикой»", [("Логистикой", "resolved", [LOGISTICS])]),
        (
            "письмо «Логистику»",
            [("Логистику", "ambiguous", [LOGISTIC, LOGISTICS])],
        ),
    ],
)
def test_names_that_share_a_case_form(
    text: str, expected: list[tuple[str, str, list[str]]]
) -> None:
    assert [summary[:3] for summary in _recognized(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Грузовик подъехал к нашему причалу ночью",
        "Причал занят до вечера",
        "дельта реки и морской порт",
        "В порт пришло судно",
        "у нас логистика хромает",
        "горизонт планирования — один год",
        "договор с «Дельта Банк»",
        "Дельта-Сервис прислал акт",
        "Транс-Порт прислал акт",
        "договор с «Кедором»",
        "Гаризантом подписан акт",
        "горизнот планирования",
    ],
)
def test_ordinary_words_and_other_names_are_not_mentions(text: str) -> None:
    assert _recognized(text) == []


@pytest.mark.parametrize(
    "organisation", ["ООО «Дельта-Банк»", "ООО «Сигма»", "АО «Кедор»", 'ПАО "Сигма"']
)
def test_named_organisation_missing_from_the_base_is_not_found(
    organisation: str,
) -> None:
    text = f"Какой срок оплаты по договору с {organisation}?"
    assert _recognized(text) == [(organisation, "not_found", [], None)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Оплата от Горизнота", [("Горизнота", "resolved", [OWN], "FUZZY")]),
        ("ООО «Гаризонт»", [("Гаризонт", "resolved", [OWN], "FUZZY")]),
        ("Делта-Порт молчит", [("Делта-Порт", "resolved", [PORT], "FUZZY")]),
        (
            "аренда у южного пирчала",
            [("южного пирчала", "resolved", [PIER], "FUZZY")],
        ),
        # Между «порта» и «портал» одна правка: опечатка не различает контрагентов.
        (
            "Делта-Портал молчит",
            [("Делта-Портал", "ambiguous", [PORT, PORTAL], "FUZZY")],
        ),
        (
            "Дельта-Портла молчит",
            [("Дельта-Портла", "ambiguous", [PORT, PORTAL], "FUZZY")],
        ),
        (
            "письмо в «Логистка»",
            [("Логистка", "ambiguous", [LOGISTIC, LOGISTICS], "FUZZY")],
        ),
    ],
)
def test_typo_is_resolved_only_with_a_margin_and_is_marked_fuzzy(
    text: str, expected: list[Summary]
) -> None:
    assert _recognized(text) == expected


@pytest.mark.parametrize(
    "spelling",
    [
        "Д-21/2031",
        "Д-21-2031",
        "д21/2031",
        "Д 21/2031",
        "D-21/2031",
        "№Д-21/2031",
        "№ Д‑21/2031",
        "Д–21/2031",
        "д-21/31",
        "Д-021/2031",
    ],
)
def test_document_number_is_resolved_in_every_spelling(spelling: str) -> None:
    text = f"Что с оплатой по договору {spelling}?"
    surface = spelling.lstrip("№ ")
    assert _recognized(text) == [(surface, "resolved", [SUPPLY], "IDENTIFIER")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("DP-04/2030", [("DP-04/2030", "resolved", [PORTAL_SUPPLY], "IDENTIFIER")]),
        ("записка OK-4", [("OK-4", "resolved", [MEMO], "IDENTIFIER")]),
        ("договор Д-21", [("Д-21", "resolved", [SUPPLY], "PARTIAL")]),
        (
            "договор Л-01",
            [("Л-01", "ambiguous", [CARRIAGE_2031, CARRIAGE_2030], "PARTIAL")],
        ),
        ("Л-01/2030", [("Л-01/2030", "resolved", [CARRIAGE_2030], "IDENTIFIER")]),
        ("Д-21/2030", [("Д-21/2030", "not_found", [SUPPLY], None)]),
        ("д 21/2030", [("д 21/2030", "not_found", [SUPPLY], None)]),
        ("Т-21/2031", [("Т-21/2031", "not_found", [], None)]),
        ("ДП-21/2031", [("ДП-21/2031", "not_found", [], None)]),
        ("ОК-4/2031", [("ОК-4/2031", "not_found", [MEMO], None)]),
    ],
)
def test_number_is_compared_by_parts_without_fuzziness(
    text: str, expected: list[Summary]
) -> None:
    assert _recognized(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "по п. 21 договора",
        "в марте 2031 года на 21 млн",
        "склад № 3",
        "срок 10-15 дней",
        "танк Т-34",
        "ИНН 6901000000",
        "счёт 69010000221",
    ],
)
def test_digits_that_name_nothing_in_the_base_are_not_mentions(text: str) -> None:
    assert _recognized(text) == []


def test_taxpayer_number_is_matched_exactly_with_the_catalog() -> None:
    assert _recognized("контрагент с ИНН 6901000022") == [
        ("6901000022", "resolved", [PORTAL], "IDENTIFIER")
    ]


def test_every_mention_gets_its_own_result() -> None:
    assert _recognized("Сравни Д-21/2031 и договор с «Дельта-Порталом»") == [
        ("Д-21/2031", "resolved", [SUPPLY], "IDENTIFIER"),
        ("Дельта-Порталом", "resolved", [PORTAL], "FULL_NAME"),
    ]


def test_number_and_name_that_disagree_are_both_reported() -> None:
    assert [summary[2] for summary in _recognized("ДП-04/2030 с «Дельта-Портом»")] == [
        [PORTAL_SUPPLY],
        [PORT],
    ]


def test_invisible_characters_do_not_hide_a_name() -> None:
    assert _recognized("«Дельта-\u200bПортал»") == [
        ("Дельта-Портал", "resolved", [PORTAL], "FULL_NAME")
    ]


def test_ambiguity_is_settled_by_a_single_candidate_named_in_another_turn() -> None:
    first = recognize("Подскажите гарантию по договору с «Дельтой».", CATALOG)
    second = recognize("«Дельта-Портал», у них оборудование покупали.", CATALOG)
    assert [_summary(mention)[:3] for mention in combine_turns([first, second])] == [
        ("Дельта-Портал", "resolved", [PORTAL])
    ]


def test_ambiguity_stays_when_both_candidates_are_named() -> None:
    first = recognize("гарантия по договору с «Дельтой»", CATALOG)
    second = recognize("«Дельта-Порт» или «Дельта-Портал»?", CATALOG)
    combined = combine_turns([first, second])
    assert [type(mention.resolution) for mention in combined] == [
        Ambiguous,
        Resolved,
        Resolved,
    ]


def test_ambiguity_stays_when_another_counterparty_is_named() -> None:
    first = recognize("гарантия по договору с «Дельтой»", CATALOG)
    second = recognize("покупатель — «Горизонт»", CATALOG)
    assert [type(m.resolution) for m in combine_turns([first, second])] == [
        Ambiguous,
        Resolved,
    ]


def test_not_found_carries_no_candidates() -> None:
    (mention,) = recognize("ООО «Сигма»", CATALOG)
    assert mention.resolution == NotFound(similar=())
