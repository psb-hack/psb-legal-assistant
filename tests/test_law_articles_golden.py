"""Разбор настоящих статей законов. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_documents import Document, LawAddress, LawArticleHeader, cite
from lex_knowledge_base import (
    KnowledgeBase,
    SourceFile,
    load_knowledge_base,
    read_archive,
)
from lex_law_lines import LawLineKind, classify_lines, split_marker
from lex_paragraphs import split_paragraphs

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "law_articles.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
# Признаки комментария справочной системы записаны здесь заново, без импорта из ядра.
SHORT_YEAR = re.compile(r"\d{4}\s+г\.")
NORM_ENDINGS = (".", ";", ":", ",", "-", "—")

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)
by_file = pytest.mark.parametrize("expected", EXPECTED, ids=lambda item: item["file"])


@functools.cache
def _base() -> KnowledgeBase:
    return load_knowledge_base(ARCHIVE)


@functools.cache
def _sources() -> tuple[SourceFile, ...]:
    return read_archive(ARCHIVE)


def _article(file_marker: str) -> tuple[SourceFile, Document, LawArticleHeader]:
    matching = [
        (source, document)
        for source, document in zip(_sources(), _base().documents, strict=True)
        if file_marker in source.name
    ]
    assert len(matching) == 1, [source.name for source, _ in matching]
    source, document = matching[0]
    assert isinstance(document.header, LawArticleHeader)
    return source, document, document.header


def _unit_addresses(document: Document, header: LawArticleHeader) -> set[LawAddress]:
    addresses = {
        LawAddress(item.number.units, None)
        for item in document.provisions
        if isinstance(item.number, LawAddress)
    }
    return addresses | {
        LawAddress(unit.address.units, None) for unit in header.repealed
    }


def test_all_thirteen_articles_are_recognised_as_laws() -> None:
    laws = [
        document
        for document in _base().documents
        if isinstance(document.header, LawArticleHeader)
    ]
    assert len(laws) == len(EXPECTED)
    assert [document.id for document in _base().documents] == [
        f"d{index:02d}" for index in range(1, len(_base().documents) + 1)
    ]


@by_file
def test_article_matches_golden(expected: dict[str, Any]) -> None:
    _, document, header = _article(expected["file"])
    assert header.citation_title == expected["cite"]
    assert header.title == expected["title"]
    assert len(document.provisions) == expected["provisions"]
    assert len(header.amendments) == expected["amendments"]
    assert len(header.comments) == expected["comments"]
    assert [unit.address.label for unit in header.repealed] == expected["repealed"]

    by_label = {item.label: item for item in document.provisions}
    assert len(by_label) == len(document.provisions)
    for label, marker in expected["markers"].items():
        assert marker in by_label[label].text
        assert cite(document, by_label[label]) == f"{expected['cite']}, {label}"


@by_file
def test_no_editorial_line_gets_into_provisions(expected: dict[str, Any]) -> None:
    _, document, _ = _article(expected["file"])
    for provision in document.provisions:
        for line in provision.text.split("\n"):
            assert SHORT_YEAR.search(line) is None, line
            assert line.endswith(NORM_ENDINGS), line


@by_file
def test_header_notes_and_provisions_together_give_the_source(
    expected: dict[str, Any],
) -> None:
    source, document, header = _article(expected["file"])
    paragraphs = split_paragraphs(source.text)
    lines = classify_lines(paragraphs)
    assert re.sub(r"\s", "", "".join(line.text for line in lines)) == re.sub(
        r"\s", "", "".join(paragraphs)
    )

    norms = []
    for line in lines:
        if line.kind is LawLineKind.NORM:
            marker = split_marker(line.text)
            norms.append(line.text if marker is None else marker.text)
    assert [
        line for item in document.provisions for line in item.text.split("\n")
    ] == norms

    outside_norms = [
        *(note.text for note in header.amendments),
        *header.comments,
        *(unit.text for unit in header.repealed),
    ]
    assert sorted(outside_norms) == sorted(
        line.text
        for line in lines
        if line.kind in (LawLineKind.COMMENT, LawLineKind.REPEALED)
    )
    source_lines = [line.text for line in lines if line.kind is LawLineKind.SOURCE]
    assert len(source_lines) == 3
    assert header.source_url in source_lines[1]
    assert header.act in source_lines[0]


@by_file
def test_every_amendment_points_at_a_unit_of_the_article(
    expected: dict[str, Any],
) -> None:
    _, document, header = _article(expected["file"])
    addresses = _unit_addresses(document, header)
    whole_article = [note for note in header.amendments if note.target is None]
    missing = [
        note.target.label
        for note in header.amendments
        if note.target is not None and note.target not in addresses
    ]
    assert missing == expected.get("missing_units", [])
    assert [note.effective_from for note in whole_article] == expected.get(
        "whole_article_amendments", []
    )


def test_probation_term_of_chief_accountant_has_one_address() -> None:
    _, document, _ = _article("ТК_ст70")
    matching = [
        item for item in document.provisions if "главных бухгалтеров" in item.text
    ]
    assert len(matching) == 1
    assert cite(document, matching[0]) == "ТК РФ, ст. 70, ч. 5"
    assert "шести месяцев" in matching[0].text
    found = _base().find(document.provision_id(matching[0]))
    assert found == (document, matching[0])
