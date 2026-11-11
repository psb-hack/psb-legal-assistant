"""Разбор настоящих документов компании. Архив в репозиторий не входит."""

from __future__ import annotations

import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_company_parsing import parse
from lex_document_header import read_header
from lex_documents import (
    DocumentHeader,
    DocumentKind,
    DocumentPart,
    DocumentStatus,
    ParsedDocument,
    Party,
)

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "company_documents.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


WITH_TREE = [item for item in EXPECTED if "sections" in item]


def _text(file_marker: str) -> str:
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = [
            name
            for name in archive.namelist()
            if file_marker in unicodedata.normalize("NFC", name)
        ]
        assert len(names) == 1, names
        return archive.read(names[0]).decode("utf-8-sig")


def _parsed(file_marker: str) -> ParsedDocument:
    return parse(_text(file_marker))


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
    header = read_header(_text(expected["file"]))
    assert header == _expected_header(expected["header"])


@pytest.mark.parametrize("expected", WITH_TREE, ids=lambda item: item["file"])
def test_document_tree_matches_golden(expected: dict[str, Any]) -> None:
    document = _parsed(expected["file"])
    body = [item for item in document.provisions if item.part is DocumentPart.BODY]

    preamble = [
        item for item in document.provisions if item.part is DocumentPart.PREAMBLE
    ]
    assert len(preamble) == expected["preamble_lines"]

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

    texts = {str(item.number): item.text for item in body}
    for number, marker in expected["markers"].items():
        assert marker in texts[number]
