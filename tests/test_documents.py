from datetime import date

from lex_documents import (
    ClauseNumber,
    Document,
    DocumentHeader,
    DocumentId,
    DocumentKind,
    DocumentPart,
    DocumentStatus,
    Heading,
    ParagraphNumber,
    Party,
    Provision,
    cite,
    context_heading,
)

CONTRACT_HEADER = DocumentHeader(
    kind=DocumentKind.CONTRACT,
    title="Договор хранения № Х-01/2030",
    number="Х-01/2030",
    dated=date(2030, 5, 5),
    parties=(
        Party("ООО «Ромашка»", "6900000001", "Хранитель"),
        Party("ООО «Василёк»", "6900000002", "Поклажедатель"),
    ),
    edition=None,
    effective_from=None,
    status=DocumentStatus.IN_FORCE,
    repealed_from=None,
)
REGULATION_HEADER = DocumentHeader(
    kind=DocumentKind.REGULATION,
    title="Регламент закупок ООО «Ромашка»",
    number=None,
    dated=date(2031, 6, 15),
    parties=(),
    edition=date(2031, 7, 1),
    effective_from=date(2031, 7, 1),
    status=DocumentStatus.IN_FORCE,
    repealed_from=None,
)
PREAMBLE_LINE = Provision(
    position=1,
    part=DocumentPart.PREAMBLE,
    number=None,
    headings=(),
    text="ДОГОВОР ХРАНЕНИЯ № Х-01/2030",
)
SUBCLAUSE = Provision(
    position=7,
    part=DocumentPart.BODY,
    number=ClauseNumber((2, 1, 2)),
    headings=(
        Heading(ClauseNumber((2,)), "Обязанности Хранителя"),
        Heading(ClauseNumber((2, 1)), "Хранитель обязан:"),
    ),
    text="вернуть оборудование по первому требованию.",
)
PARAGRAPH = Provision(
    position=5,
    part=DocumentPart.BODY,
    number=ParagraphNumber(3),
    headings=(),
    text="Заявки подаются до 1 марта.",
)
CONTRACT = Document(DocumentId("d03"), CONTRACT_HEADER, (PREAMBLE_LINE, SUBCLAUSE))
REGULATION = Document(DocumentId("d11"), REGULATION_HEADER, (PARAGRAPH,))


def test_clause_is_labelled_by_its_number() -> None:
    assert SUBCLAUSE.label == "п. 2.1.2"


def test_paragraph_is_labelled_by_its_position_in_body_text() -> None:
    assert PARAGRAPH.label == "абз. 3"


def test_line_before_first_clause_is_labelled_as_preamble() -> None:
    assert PREAMBLE_LINE.label == "преамбула"


def test_provision_id_joins_document_and_position() -> None:
    assert CONTRACT.provision_id(SUBCLAUSE) == "d03.7"


def test_citation_names_document_and_provision() -> None:
    assert cite(CONTRACT, SUBCLAUSE) == "Договор хранения № Х-01/2030, п. 2.1.2"
    assert cite(CONTRACT, PREAMBLE_LINE) == "Договор хранения № Х-01/2030, преамбула"


def test_citation_of_regulation_names_its_edition() -> None:
    assert cite(REGULATION, PARAGRAPH) == (
        "Регламент закупок ООО «Ромашка» в редакции от 01.07.2031, абз. 3"
    )


def test_context_heading_lists_parties_and_path_of_headings() -> None:
    assert context_heading(CONTRACT, SUBCLAUSE) == (
        "Договор хранения № Х-01/2030 (ООО «Ромашка», ООО «Василёк»), "
        "раздел 2 «Обязанности Хранителя», п. 2.1 «Хранитель обязан:», п. 2.1.2"
    )


def test_context_heading_of_document_without_parties_has_no_brackets() -> None:
    assert context_heading(REGULATION, PARAGRAPH) == (
        "Регламент закупок ООО «Ромашка» в редакции от 01.07.2031, абз. 3"
    )
