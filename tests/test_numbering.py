from lex_documents import ClauseNumber
from lex_numbering import split_numbered_line


def test_clause_line_splits_into_number_and_text() -> None:
    line = split_numbered_line("3.1. Вознаграждение составляет 70 000 рублей.")
    assert line is not None
    assert line.number == ClauseNumber((3, 1))
    assert line.text == "Вознаграждение составляет 70 000 рублей."


def test_section_heading_line_has_single_level_number() -> None:
    line = split_numbered_line("10. Реквизиты и подписи сторон")
    assert line is not None
    assert line.number == ClauseNumber((10,))
    assert line.text == "Реквизиты и подписи сторон"


def test_date_at_line_start_is_not_a_clause_number() -> None:
    assert split_numbered_line("01.03.2030. Редакция вступает в силу.") is None
    assert split_numbered_line("15.11.2030. Письмо направлено адресату.") is None


def test_number_inside_line_is_not_a_clause() -> None:
    line = "Приложение 1. Перечень проверок: 1. Проверка лица. 2. Проверка банка."
    assert split_numbered_line(line) is None


def test_quoted_clause_of_another_document_is_not_a_clause() -> None:
    assert split_numbered_line("«3.2. Товар доставляется по новому адресу».") is None


def test_number_without_trailing_dot_is_not_a_clause() -> None:
    assert split_numbered_line("2030 год объявлен годом архива.") is None


def test_clause_number_knows_its_descendants() -> None:
    section = ClauseNumber((2,))
    assert section.is_ancestor_of(ClauseNumber((2, 1)))
    assert section.is_ancestor_of(ClauseNumber((2, 1, 3)))
    assert not section.is_ancestor_of(ClauseNumber((2,)))
    assert not section.is_ancestor_of(ClauseNumber((20, 1)))
    assert str(ClauseNumber((2, 1, 3))) == "2.1.3"
