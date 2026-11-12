from datetime import date

import pytest

from lex_document_graph import DocumentGraph, Edition, Link, build_graph
from lex_document_links import LinkKind
from lex_documents import ClauseNumber, DocumentId, ProvisionId
from synthetic_base import LAW_ARTICLE, MEMO
from synthetic_links import (
    CONTRACT,
    FROM_SIGNING,
    NEW_RULES,
    OLD_RULES,
    agreement,
    knowledge_base,
    letter,
)

D01, D02, D03 = DocumentId("d01"), DocumentId("d02"), DocumentId("d03")
OTHER_CONTRACT = CONTRACT.replace("С-21/2031", "С-22/2031")
OLD_RULES_WITHOUT_REPEAL_NOTE = OLD_RULES.split("\n", 1)[1]


def _graph(*texts: str) -> DocumentGraph:
    return build_graph(knowledge_base(*texts))


def _only_link(graph: DocumentGraph) -> Link:
    (link,) = graph.links
    return link


def test_supplementary_agreement_is_found_from_its_contract() -> None:
    graph = _graph(CONTRACT, agreement(FROM_SIGNING))
    assert graph.amendments_of(D01) == (D02,)
    assert graph.links == (
        Link(
            kind=LinkKind.AMENDS,
            source=D02,
            stated_in=ProvisionId("d02.2"),
            reference="договору подряда № С-21/2031 от 5 мая 2031 г.",
            target=D01,
            clause=None,
            target_provision=None,
            effective_from=date(2031, 6, 10),
        ),
    )
    assert graph.links_from(D02) == graph.links_to(D01) == graph.links
    assert graph.unresolved == ()


@pytest.mark.parametrize(
    "parent",
    [
        "к договору подряда № с21/2031 от 5 мая 2031 г.",
        "к договору подряда № C-21-2031",
        "к договору подряда от 5 мая 2031 г.",
    ],
)
def test_contract_is_found_by_number_in_any_spelling_or_by_title(parent: str) -> None:
    graph = _graph(CONTRACT, agreement(parent=parent))
    assert graph.amendments_of(D01) == (D02,)


@pytest.mark.parametrize(
    ("parent", "contracts"),
    [
        # Договора с таким годом в базе нет.
        ("к договору подряда № С-21/2030 от 5 мая 2030 г.", (CONTRACT,)),
        # Под название без номера подходят оба договора.
        ("к договору подряда от 5 мая 2031 г.", (CONTRACT, OTHER_CONTRACT)),
        # Одного слова мало, чтобы назвать документ.
        ("к договору от 5 мая 2031 г.", (CONTRACT,)),
    ],
)
def test_link_without_a_single_target_stays_unresolved(
    parent: str, contracts: tuple[str, ...]
) -> None:
    graph = _graph(agreement(parent=parent), *contracts)
    link = _only_link(graph)
    assert link.target is None
    assert link.reference == parent.removeprefix("к ")
    assert graph.unresolved == (link,)
    assert graph.amendments_of(D02) == ()


def test_editions_of_one_document_form_a_family_with_periods() -> None:
    graph = _graph(NEW_RULES, CONTRACT, OLD_RULES)
    family = graph.family_of(D01)
    assert family is not None
    assert family is graph.family_of(D03)
    assert graph.families == (family,)
    assert family.key == 'regulation:положение о закупках ооо "бирюза"'
    assert family.title == "Положение о закупках ООО «Бирюза»"
    assert family.editions == (
        Edition(D03, date(2029, 2, 2), date(2029, 2, 2), date(2031, 7, 15)),
        Edition(D01, date(2031, 7, 1), date(2031, 7, 15), None),
    )
    assert graph.family_of(D02) is None


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        (date(2029, 2, 1), None),
        (date(2029, 2, 2), D02),
        (date(2031, 7, 14), D02),
        (date(2031, 7, 15), D01),
        (date(2040, 1, 1), D01),
    ],
)
def test_edition_at_respects_the_boundary_of_periods(
    on: date, expected: DocumentId | None
) -> None:
    family = _graph(NEW_RULES, OLD_RULES).families[0]
    edition = family.edition_at(on)
    assert (None if edition is None else edition.document) == expected


def test_edition_without_repeal_note_ends_when_the_next_one_starts() -> None:
    family = _graph(OLD_RULES_WITHOUT_REPEAL_NOTE, NEW_RULES).families[0]
    assert family.editions[0].effective_to == date(2031, 7, 15)


def test_gap_between_editions_has_no_edition() -> None:
    earlier_repeal = OLD_RULES.replace("с 15.07.2031", "с 01.07.2031")
    family = _graph(earlier_repeal, NEW_RULES).families[0]
    assert family.edition_at(date(2031, 7, 10)) is None


def test_new_edition_links_to_the_edition_it_repeals() -> None:
    link = _only_link(_graph(OLD_RULES, NEW_RULES))
    assert (link.kind, link.source, link.target) == (LinkKind.SUPERSEDES, D02, D01)
    assert link.stated_in == ProvisionId("d02.3")
    assert link.effective_from == date(2031, 7, 15)


def test_repealed_edition_missing_from_the_base_is_an_unresolved_link() -> None:
    graph = _graph(NEW_RULES)
    link = _only_link(graph)
    assert link.target is None
    assert link.reference == "Редакция от 02.02.2029 утрачивает силу"
    assert graph.unresolved == (link,)
    assert graph.families[0].editions == (
        Edition(D01, date(2031, 7, 1), date(2031, 7, 15), None),
    )


def test_letter_supplements_a_clause_of_the_named_edition() -> None:
    graph = _graph(OLD_RULES, NEW_RULES, letter())
    (link,) = graph.links_from(D03)
    assert link == Link(
        kind=LinkKind.SUPPLEMENTS,
        source=D03,
        stated_in=ProvisionId("d03.6"),
        reference="п. 2.1 Положения о закупках (ред. от 01.07.2031)",
        target=D02,
        clause=ClauseNumber((2, 1)),
        target_provision=ProvisionId("d02.5"),
        effective_from=date(2031, 9, 1),
    )
    assert link.is_resolved


@pytest.mark.parametrize(
    ("dated", "expected"), [("20.08.2031", D02), ("20.08.2030", D01)]
)
def test_reference_without_edition_means_the_edition_in_force_at_source_date(
    dated: str, expected: DocumentId
) -> None:
    sentence = "Проверка идёт в дополнение к п. 2.1 Положения о закупках."
    graph = _graph(OLD_RULES, NEW_RULES, letter(sentence, dated))
    assert graph.links_from(D03)[0].target == expected


@pytest.mark.parametrize(
    ("sentence", "target"),
    [
        # Такой редакции в базе нет.
        ("В дополнение к п. 2.1 Положения о закупках (ред. от 05.05.2030).", None),
        # Одного слова мало, чтобы назвать документ.
        ("В дополнение к п. 2.1 Положения.", None),
        # Документ найден, пункта в нём нет.
        ("В дополнение к п. 9.9 Положения о закупках (ред. от 01.07.2031).", D02),
    ],
)
def test_supplement_with_missing_target_or_clause_is_unresolved(
    sentence: str, target: DocumentId | None
) -> None:
    graph = _graph(OLD_RULES, NEW_RULES, letter(sentence))
    (link,) = graph.links_from(D03)
    assert link.target == target
    assert link.target_provision is None
    assert link in graph.unresolved


def test_documents_that_name_no_other_document_have_no_links() -> None:
    graph = _graph(CONTRACT, MEMO, LAW_ARTICLE, OTHER_CONTRACT)
    assert graph == DocumentGraph(links=(), families=())
    assert graph.amendments_of(D01) == ()
    assert graph.family_of(D02) is None
