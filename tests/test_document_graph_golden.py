"""Граф связей на настоящей базе. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from lex_document_graph import DocumentGraph, build_graph
from lex_documents import DocumentHeader, DocumentId, ProvisionId, cite
from lex_knowledge_base import KnowledgeBase, load_knowledge_base

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "document_graph.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _base() -> KnowledgeBase:
    return load_knowledge_base(ARCHIVE)


@functools.cache
def _graph() -> DocumentGraph:
    return build_graph(_base())


def _title(document_id: DocumentId) -> str:
    (document,) = (item for item in _base().documents if item.id == document_id)
    return document.header.citation_title


def _label(provision_id: ProvisionId | None) -> str | None:
    found = None if provision_id is None else _base().find(provision_id)
    return None if found is None else found[1].label


def test_links_are_exactly_those_stated_in_the_texts() -> None:
    actual = [
        {
            "kind": link.kind.value,
            "source": _title(link.source),
            "stated_in": _label(link.stated_in),
            "reference": link.reference,
            "target": None if link.target is None else _title(link.target),
            "target_provision": _label(link.target_provision),
            "effective_from": link.effective_from,
        }
        for link in _graph().links
    ]
    assert actual == EXPECTED["links"]
    assert _graph().unresolved == ()


def test_editions_of_the_regulation_form_one_family() -> None:
    actual = [
        {
            "title": family.title,
            "editions": [
                {
                    "dated": edition.dated,
                    "effective_from": edition.effective_from,
                    "effective_to": edition.effective_to,
                }
                for edition in family.editions
            ],
        }
        for family in _graph().families
    ]
    assert actual == EXPECTED["families"]


def test_documents_without_links_stay_unlinked() -> None:
    unlinked = [
        document.header.citation_title
        for document in _base().documents
        if isinstance(document.header, DocumentHeader)
        and not _graph().links_from(document.id)
        and not _graph().links_to(document.id)
    ]
    assert unlinked == EXPECTED["unlinked"]


def test_supplementary_agreement_is_found_from_the_contract() -> None:
    amended = EXPECTED["links"][0]
    (contract,) = (
        item for item in _base().documents if _title(item.id) == amended["target"]
    )
    amendments = _graph().amendments_of(contract.id)
    assert [_title(document_id) for document_id in amendments] == [amended["source"]]


@pytest.mark.parametrize("case", EXPECTED["edition_at"], ids=str)
def test_edition_of_the_regulation_on_a_date(case: dict[str, object]) -> None:
    (family,) = _graph().families
    edition = family.edition_at(case["date"])
    assert (None if edition is None else edition.dated) == case["dated"]


def test_cited_target_of_the_letter_is_the_clause_of_the_current_edition() -> None:
    (link,) = (item for item in _graph().links if item.target_provision is not None)
    found = _base().find(link.target_provision)
    assert found is not None
    assert cite(*found) == (
        "Регламент согласования договоров ООО «Меридиан» "
        "в редакции от 01.03.2025, п. 5.4"
    )
