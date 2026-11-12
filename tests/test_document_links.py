from datetime import date

import pytest

from lex_document_links import LinkKind, LinkStatement, find_link_statements
from lex_documents import ClauseNumber
from synthetic_base import LAW_ARTICLE, MEMO
from synthetic_links import (
    CONTRACT,
    FROM_SIGNING,
    NEW_RULES,
    OLD_RULES,
    agreement,
    document,
    letter,
)


def _statements(text: str) -> tuple[LinkStatement, ...]:
    return find_link_statements(document(text))


def _only(text: str) -> LinkStatement:
    (statement,) = _statements(text)
    return statement


def test_supplementary_agreement_names_its_contract_in_the_title() -> None:
    statement = _only(agreement(FROM_SIGNING))
    assert statement.kind is LinkKind.AMENDS
    assert statement.stated_in.position == 2
    assert statement.reference.text == "договору подряда № С-21/2031 от 5 мая 2031 г."
    assert statement.reference.title_words == ("договору", "подряда")
    assert statement.reference.number == "С-21/2031"
    assert statement.clause is None


def test_parent_document_without_number_is_named_by_title() -> None:
    reference = _only(agreement(parent="к договору подряда от 5 мая 2031 г.")).reference
    assert reference.number is None
    assert reference.title_words[:2] == ("договору", "подряда")


def test_title_line_of_another_genre_does_not_name_a_parent_document() -> None:
    memo = MEMO.replace("О графике отпусков", "к договору подряда № С-21/2031")
    assert _statements(memo) == ()


def test_new_edition_names_the_edition_it_repeals() -> None:
    statement = _only(NEW_RULES)
    assert statement.kind is LinkKind.SUPERSEDES
    assert statement.stated_in.position == 3
    assert statement.reference.text == "Редакция от 02.02.2029 утрачивает силу"
    assert statement.reference.edition == date(2029, 2, 2)
    assert statement.reference.title_words == ()
    assert statement.effective_from == date(2031, 7, 15)


@pytest.mark.parametrize(
    "text",
    [
        OLD_RULES,
        OLD_RULES.replace(
            "РЕДАКЦИЯ УТРАТИЛА СИЛУ с 15.07.2031",
            "Редакция от 02.02.2029 утратила силу с 15.07.2031",
        ),
    ],
)
def test_note_about_own_repeal_is_not_a_link(text: str) -> None:
    assert _statements(text) == ()


def test_supplement_names_clause_title_and_edition() -> None:
    statement = _only(letter())
    assert statement.kind is LinkKind.SUPPLEMENTS
    assert statement.stated_in.label == "абз. 2"
    assert (
        statement.reference.text == "п. 2.1 Положения о закупках (ред. от 01.07.2031)"
    )
    assert statement.reference.title_words == ("положения", "о", "закупках")
    assert statement.reference.edition == date(2031, 7, 1)
    assert statement.clause == ClauseNumber((2, 1))
    assert statement.effective_from == date(2031, 9, 1)


@pytest.mark.parametrize(
    ("sentence", "clause", "words", "number", "edition"),
    [
        (
            "Порядок применяется в дополнение к Положению о закупках "
            "в редакции от 1 июля 2031 г.",
            None,
            ("положению", "о", "закупках"),
            None,
            date(2031, 7, 1),
        ),
        (
            "В дополнение к пункту 4 Договора подряда № С-21/2031 согласован график.",
            ClauseNumber((4,)),
            ("договора", "подряда"),
            "С-21/2031",
            None,
        ),
        (
            "Проверка идёт в дополнение к п. 2.1 Положения о закупках, до его замены.",
            ClauseNumber((2, 1)),
            ("положения", "о", "закупках"),
            None,
            None,
        ),
    ],
)
def test_supplement_phrasings(
    sentence: str,
    clause: ClauseNumber | None,
    words: tuple[str, ...],
    number: str | None,
    edition: date | None,
) -> None:
    statement = _only(letter(sentence))
    assert statement.clause == clause
    assert statement.reference.title_words == words
    assert statement.reference.number == number
    assert statement.reference.edition == edition


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ((FROM_SIGNING,), date(2031, 6, 10)),
        (
            ("Соглашение вступает в силу с момента его подписания Сторонами.",),
            date(2031, 6, 10),
        ),
        (("Соглашение вступает в силу с 1 августа 2031 г.",), date(2031, 8, 1)),
        (("Новый срок применяется с 01.08.2031.",), date(2031, 8, 1)),
        (("С 1 августа 2031 г. работы ведутся в две смены.",), date(2031, 8, 1)),
        ((FROM_SIGNING, "С 1 августа 2031 г. работы ведутся в две смены."), None),
        (("Срок выполнения работ продлевается на 10 рабочих дней.",), None),
    ],
)
def test_commencement_date_of_the_source(
    body: tuple[str, ...], expected: date | None
) -> None:
    assert _only(agreement(*body)).effective_from == expected


@pytest.mark.parametrize("text", [CONTRACT, MEMO, LAW_ARTICLE])
def test_document_that_names_no_other_document_has_no_statements(text: str) -> None:
    assert _statements(text) == ()
