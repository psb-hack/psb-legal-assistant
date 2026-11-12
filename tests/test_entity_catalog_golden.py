"""Каталог сущностей на настоящей базе. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from lex_document_numbers import DocumentNumber
from lex_entity_catalog import EntityCatalog, build_catalog
from lex_knowledge_base import load_knowledge_base

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "entity_catalog.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _catalog() -> EntityCatalog:
    return build_catalog(load_knowledge_base(ARCHIVE))


def test_counterparties_match_the_headers_of_the_base() -> None:
    titles = {entry.id: entry.title for entry in _catalog().documents}
    actual = [
        {
            "name": item.name,
            "taxpayer_number": item.taxpayer_number,
            "roles": list(item.roles),
            "documents": [titles[document_id] for document_id in item.documents],
            "name_tokens": {
                token.nominative: "common" if token.common else "distinctive"
                for token in item.name_tokens
            },
        }
        for item in _catalog().counterparties
    ]
    assert actual == EXPECTED["counterparties"]


def test_documents_match_the_headers_of_the_base() -> None:
    actual = [
        (entry.title, entry.kind.value, entry.number, entry.edition)
        for entry in _catalog().documents
    ]
    assert actual == [
        (
            expected["title"],
            expected["kind"],
            _number(expected.get("number")),
            expected.get("edition"),
        )
        for expected in EXPECTED["documents"]
    ]


def _number(parts: dict[str, object] | None) -> DocumentNumber | None:
    if parts is None:
        return None
    return DocumentNumber(parts["prefix"], parts["number"], parts.get("year"))
