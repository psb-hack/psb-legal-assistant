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
from lex_documents import DocumentPart, ParsedDocument

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "company_documents.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


def _parsed(file_marker: str) -> ParsedDocument:
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = [
            name
            for name in archive.namelist()
            if file_marker in unicodedata.normalize("NFC", name)
        ]
        assert len(names) == 1, names
        return parse(archive.read(names[0]).decode("utf-8-sig"))


@pytest.mark.parametrize("expected", EXPECTED, ids=lambda item: item["file"])
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
