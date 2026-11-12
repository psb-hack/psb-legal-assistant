from lex_company_parsing import parse
from lex_documents import (
    ClauseNumber,
    DocumentPart,
    Heading,
    ParsedDocument,
    Provision,
    cite,
)

STORAGE_CONTRACT = """\
ДОГОВОР ХРАНЕНИЯ № Х-01/2030
г. Тверь — 5 мая 2030 г.
ООО «Ромашка» (ИНН 6900000001), далее — «Хранитель», с одной стороны, и
ООО «Василёк» (ИНН 6900000002), далее — «Поклажедатель», с другой стороны,
заключили настоящий договор о нижеследующем.
1. Предмет договора
1.1. Хранитель принимает на хранение оборудование Поклажедателя.
1.2. Срок хранения — 12 месяцев.
2. Обязанности Хранителя
2.1. Хранитель обязан:
2.1.1. обеспечить сохранность оборудования;
2.1.2. вернуть оборудование по первому требованию.
2.2. Вознаграждение составляет 70 000 рублей в месяц
и выплачивается до 10-го числа.
3. Реквизиты и подписи сторон
Хранитель: ООО «Ромашка», г. Тверь.
Поклажедатель: ООО «Василёк», г. Тверь.
"""


def _in_part(document: ParsedDocument, part: DocumentPart) -> list[Provision]:
    return [item for item in document.provisions if item.part is part]


def _clause(document: ParsedDocument, *parts: int) -> Provision:
    found = [item for item in document.provisions if item.number == ClauseNumber(parts)]
    assert len(found) == 1
    return found[0]


def test_lines_before_first_number_form_preamble() -> None:
    preamble = _in_part(parse(STORAGE_CONTRACT), DocumentPart.PREAMBLE)
    assert len(preamble) == 5
    assert preamble[0].text == "ДОГОВОР ХРАНЕНИЯ № Х-01/2030"
    assert all(item.number is None for item in preamble)


def test_numbered_lines_become_body_provisions() -> None:
    body = _in_part(parse(STORAGE_CONTRACT), DocumentPart.BODY)
    assert [str(item.number) for item in body] == [
        "1.1",
        "1.2",
        "2.1",
        "2.1.1",
        "2.1.2",
        "2.2",
    ]


def test_provision_keeps_its_section_heading() -> None:
    clause = _clause(parse(STORAGE_CONTRACT), 1, 2)
    assert clause.text == "Срок хранения — 12 месяцев."
    assert clause.headings == (Heading(ClauseNumber((1,)), "Предмет договора"),)


def test_section_heading_is_not_a_provision() -> None:
    numbers = {item.number for item in parse(STORAGE_CONTRACT).provisions}
    assert ClauseNumber((1,)) not in numbers
    assert ClauseNumber((2,)) not in numbers


def test_lead_in_clause_is_provision_and_heading_of_its_subclauses() -> None:
    document = parse(STORAGE_CONTRACT)
    assert _clause(document, 2, 1).text == "Хранитель обязан:"
    assert _clause(document, 2, 1, 2).headings == (
        Heading(ClauseNumber((2,)), "Обязанности Хранителя"),
        Heading(ClauseNumber((2, 1)), "Хранитель обязан:"),
    )


def test_unnumbered_line_continues_previous_provision() -> None:
    clause = _clause(parse(STORAGE_CONTRACT), 2, 2)
    assert clause.text == (
        "Вознаграждение составляет 70 000 рублей в месяц\n"
        "и выплачивается до 10-го числа."
    )
    assert clause.headings == (Heading(ClauseNumber((2,)), "Обязанности Хранителя"),)


def test_requisites_section_is_marked_as_requisites() -> None:
    requisites = _in_part(parse(STORAGE_CONTRACT), DocumentPart.REQUISITES)
    assert len(requisites) == 1
    assert requisites[0].number == ClauseNumber((3,))
    assert "Поклажедатель: ООО «Василёк»" in requisites[0].text


def test_clause_mentioning_requisites_stays_in_body() -> None:
    document = parse("1. Прочее\n1.1. Реквизиты сторон меняются уведомлением.\n")
    assert _clause(document, 1, 1).part is DocumentPart.BODY


def test_top_level_clause_without_subclauses_is_a_provision() -> None:
    document = parse(
        "1. Предмет\n1.1. Услуги оказываются.\n2. Договор составлен в двух копиях.\n"
    )
    clause = _clause(document, 2)
    assert clause.part is DocumentPart.BODY
    assert clause.headings == ()


def test_no_line_of_source_text_is_lost() -> None:
    document = parse(STORAGE_CONTRACT)
    kept = "\n".join(item.text for item in document.provisions)
    kept += "\n" + "\n".join(
        heading.title for item in document.provisions for heading in item.headings
    )
    for line in STORAGE_CONTRACT.splitlines():
        text = line.split(". ", 1)[1] if line[0].isdigit() else line
        assert text in kept


def test_bom_crlf_and_blank_lines_do_not_change_result() -> None:
    noisy = "﻿" + STORAGE_CONTRACT.replace("\n", "\r\n\r\n")
    assert parse(noisy) == parse(STORAGE_CONTRACT)


def test_citation_names_document_and_clause() -> None:
    document = parse(STORAGE_CONTRACT)
    title = "Договор хранения № Х-01/2030"
    assert (
        cite(title, _clause(document, 2, 2)) == "Договор хранения № Х-01/2030, п. 2.2"
    )
    assert cite(title, document.provisions[0]) == (
        "Договор хранения № Х-01/2030, преамбула"
    )
