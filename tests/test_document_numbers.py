import pytest

from lex_document_numbers import (
    DocumentNumber,
    find_document_numbers,
    parse_document_number,
)

P17 = DocumentNumber("п", 17, 2024)


@pytest.mark.parametrize(
    ("header_number", "expected"),
    [
        ("П-17/2024", P17),
        ("ПТ-05/2023", DocumentNumber("пт", 5, 2023)),
        ("ЮО-7", DocumentNumber("юо", 7, None)),
        ("К-12", DocumentNumber("к", 12, None)),
        ("1", None),
        ("14-П", None),
        ("2031/17-ПС", None),
    ],
)
def test_header_number_is_split_into_parts(
    header_number: str, expected: DocumentNumber | None
) -> None:
    assert parse_document_number(header_number) == expected


@pytest.mark.parametrize(
    "text",
    [
        "П-17/2024",
        "П-17-2024",
        "п17/2024",
        "П 17/2024",
        "№П-17/2024",
        "№ П‑17/2024",
        "П–17/2024",
        "П—17/2024",
        "п-17/24",
        "П-017/2024",
        "П-17 / 2024",
        "П-17\\2024",
        "по договору (П-17/2024).",
        "Пользователь: П-17/2024. Доставка включена?",
        "договор\u00a0П-17/2024",
    ],
)
def test_one_number_is_found_in_every_spelling(text: str) -> None:
    assert [found.number for found in find_document_numbers(text)] == [P17]


@pytest.mark.parametrize(
    ("text", "catalogued"),
    [
        ("P-17/2024", "П-17/2024"),  # латинская «P» вместо «П»: по звучанию
        ("PT-05/2023", "ПТ-05/2023"),
        ("A-08/2024", "А-08/2024"),  # латинская «A»: по начертанию
        ("K-12", "К-12"),
        ("п-17", "П-17/2024"),
        ("П-17/2023", "П-17/2024"),
    ],
)
def test_prefix_is_compared_in_every_reading(text: str, catalogued: str) -> None:
    (found,) = find_document_numbers(text)
    expected = parse_document_number(catalogued)
    assert expected is not None
    assert found.number.same_series(expected)


@pytest.mark.parametrize(
    "other", ["Т-17/2024", "АП-17/2024", "ПТ-17/2024", "П-171/2024", "П-71/2024"]
)
def test_other_prefix_or_number_is_another_series(other: str) -> None:
    parsed = parse_document_number(other)
    assert parsed is not None
    assert not parsed.same_series(P17)


def test_latin_p_reads_both_as_cyrillic_er_and_pe() -> None:
    (found,) = find_document_numbers("P-17/2024")
    assert found.number.same_series(DocumentNumber("р", 17, 2024))
    assert found.number.same_series(P17)


@pytest.mark.parametrize(
    "text",
    [
        "по п. 17 договора",
        "в августе 2024 года на 12 млн",
        "ст. 13.11 КоАП",
        "склад № 3",
        "от 12.02.2024",
        "срок 10-15 дней",
        "работаем с 8-24",
        "а 8/24 не подойдёт",
        "П-17/20245",
        "П-17/2024-1",
        "СП-17/2024а",
        "приказ № 14-П",
        "п 17",
        "п17",
    ],
)
def test_ordinary_text_is_not_a_number(text: str) -> None:
    assert find_document_numbers(text) == ()


def test_several_numbers_keep_their_order_and_positions() -> None:
    text = "сравни П-17/2024 и ПТ-05/2023, а также ЮО-7"
    found = find_document_numbers(text)
    assert [text[item.start : item.end] for item in found] == [
        "П-17/2024",
        "ПТ-05/2023",
        "ЮО-7",
    ]
    assert found[2].number == DocumentNumber("юо", 7, None)


@pytest.mark.parametrize(
    ("text", "written_in_full"),
    [
        ("П-17/2023", True),
        ("п17-2023", True),
        ("П 17/2023", False),
        ("П-17 / 2023", False),
        ("п-17/23", False),
        ("П-17", False),
    ],
)
def test_only_compact_spelling_with_full_year_names_a_document(
    text: str, written_in_full: bool
) -> None:
    (found,) = find_document_numbers(text)
    assert found.written_in_full is written_in_full
