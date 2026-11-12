"""Выбор редакции по дате события на настоящей базе. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_dates import PartialDate, find_dates
from lex_document_graph import EditionFamily, build_graph
from lex_documents import Document, LawAddress, LawArticleHeader
from lex_edition_choice import (
    EditionChoice,
    EditionInForce,
    EditionsAcrossBoundary,
    NoEditionOnDate,
    current_edition,
    edition_on,
)
from lex_event_date import EventDate, read_event_date
from lex_knowledge_base import KnowledgeBase, load_knowledge_base
from lex_law_validity import (
    LawTextInForce,
    LawTextNewer,
    LawTextOnDate,
    law_text_as_retrieved,
    law_text_on,
)

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "edition_choice.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
TODAY = EXPECTED["today"]

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _base() -> KnowledgeBase:
    return load_knowledge_base(ARCHIVE)


@functools.cache
def _family() -> EditionFamily:
    (family,) = build_graph(_base()).families
    return family


def _document(title: str) -> Document:
    (document,) = (
        item for item in _base().documents if item.header.citation_title == title
    )
    return document


def _choice(question: str) -> EditionChoice:
    event = read_event_date(question, TODAY)
    if event is None:
        return current_edition(_family(), TODAY, _base())
    assert isinstance(event, EventDate)
    return edition_on(_family(), event.period, _base())


def _described(choice: EditionChoice) -> dict[str, Any]:
    if isinstance(choice, EditionInForce):
        return {"edition": choice.edition.dated, "date_stated": choice.date_stated}
    if isinstance(choice, EditionsAcrossBoundary):
        return {
            "across": [edition.dated for edition in choice.editions],
            "boundaries": list(choice.boundaries),
        }
    assert isinstance(choice, NoEditionOnDate)
    return {"missing": choice.reason.value}


@pytest.mark.parametrize("case", EXPECTED["company"], ids=lambda case: case["question"])
def test_edition_of_the_regulation_for_the_question(case: dict[str, Any]) -> None:
    choice = _choice(case["question"])
    expected = {
        key: value
        for key, value in case.items()
        if key not in ("question", "clause", "states")
    }
    assert _described(choice) == expected


@pytest.mark.parametrize(
    "case",
    [case for case in EXPECTED["company"] if "clause" in case],
    ids=lambda case: case["question"],
)
def test_chosen_edition_holds_the_provision_with_the_threshold(
    case: dict[str, Any],
) -> None:
    choice = _choice(case["question"])
    assert isinstance(choice, EditionInForce)
    (document,) = (
        item for item in _base().documents if item.id == choice.edition.document
    )
    (provision,) = (
        item for item in document.provisions if item.label == case["clause"]
    )
    assert provision.text == case["states"]


def _law_outcome(case: dict[str, Any]) -> LawTextOnDate:
    document = _document(case["article"])
    header = document.header
    assert isinstance(header, LawArticleHeader)
    address = None
    if case["provision"] is not None:
        (address,) = (
            item.number
            for item in document.provisions
            if item.label == case["provision"]
        )
        assert isinstance(address, LawAddress)
    if case["event"] is None:
        return law_text_as_retrieved(header, address)
    (mention,) = find_dates(case["event"])
    period: PartialDate = mention.period
    return law_text_on(header, address, period)


@pytest.mark.parametrize(
    "case",
    EXPECTED["laws"],
    ids=lambda case: f"{case['article']}, {case['provision']}, {case['event']}",
)
def test_law_text_on_the_event_date(case: dict[str, Any]) -> None:
    outcome = _law_outcome(case)
    if isinstance(outcome, LawTextInForce):
        actual = {"outcome": "in_force", "note": outcome.amended}
        assert outcome.date_stated is (case["event"] is not None)
    elif isinstance(outcome, LawTextNewer):
        actual = {"outcome": "newer", "note": outcome.amended}
    else:
        actual = {
            "outcome": "uncertain",
            "reason": outcome.reason.value,
            "note": outcome.note,
        }
    note = actual["note"]
    actual["note"] = None if note is None else note.effective_from
    expected = {
        key: value
        for key, value in case.items()
        if key not in ("article", "provision", "event")
    }
    assert actual == expected
