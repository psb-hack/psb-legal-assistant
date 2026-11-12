"""Разбор настоящих документов компании. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_documents import (
    Document,
    DocumentHeader,
    DocumentId,
    DocumentKind,
    DocumentPart,
    DocumentStatus,
    Party,
    cite,
)
from lex_knowledge_base import SourceFile, company_document, read_archive

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "company_documents.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
WITH_SECTIONS = [item for item in EXPECTED if "sections" in item]

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _sources() -> tuple[SourceFile, ...]:
    return read_archive(ARCHIVE)


def _document(file_marker: str) -> Document:
    matching = [source for source in _sources() if file_marker in source.name]
    assert len(matching) == 1, [source.name for source in matching]
    return company_document(DocumentId("d01"), matching[0])


def _expected_header(header: dict[str, Any]) -> DocumentHeader:
    return DocumentHeader(
        kind=DocumentKind(header["kind"]),
        title=header["title"],
        number=header["number"],
        dated=header["dated"],
        parties=tuple(Party(**party) for party in header["parties"]),
        edition=header["edition"],
        effective_from=header["effective_from"],
        status=DocumentStatus(header["status"]),
        repealed_from=header["repealed_from"],
    )


@pytest.mark.parametrize("expected", EXPECTED, ids=lambda item: item["file"])
def test_document_header_matches_golden(expected: dict[str, Any]) -> None:
    document = _document(expected["file"])
    assert document.header == _expected_header(expected["header"])


@pytest.mark.parametrize("expected", EXPECTED, ids=lambda item: item["file"])
def test_provisions_match_golden(expected: dict[str, Any]) -> None:
    document = _document(expected["file"])
    assert len(document.provisions) == expected["provisions"]

    preamble = [
        item for item in document.provisions if item.part is DocumentPart.PREAMBLE
    ]
    assert len(preamble) == expected["preamble_lines"]

    addressed = {
        item.label: item for item in document.provisions if item.number is not None
    }
    for label, marker in expected["markers"].items():
        assert marker in addressed[label].text
    for label, citation in expected["cites"].items():
        assert cite(document, addressed[label]) == citation


@pytest.mark.parametrize("expected", WITH_SECTIONS, ids=lambda item: item["file"])
def test_sections_match_golden(expected: dict[str, Any]) -> None:
    document = _document(expected["file"])
    body = [item for item in document.provisions if item.part is DocumentPart.BODY]
    sections = Counter(
        (str(item.headings[0].number), item.headings[0].title) for item in body
    )
    assert sections == {
        (section["number"], section["title"]): section["clauses"]
        for section in expected["sections"]
    }

    requisites = [
        str(item.number)
        for item in document.provisions
        if item.part is DocumentPart.REQUISITES
    ]
    assert requisites == ([expected["requisites"]] if expected["requisites"] else [])
