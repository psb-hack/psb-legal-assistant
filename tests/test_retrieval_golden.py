"""Цифры поиска на настоящей базе и golden-наборе. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_entity_catalog import build_catalog
from lex_knowledge_base import load_knowledge_base
from lex_retrieval import LexicalRetriever
from lex_sanitizer import admitted_provisions, default_detectors, screen
from lex_settings import SanitizerThresholds
from retrieval_eval import evaluate, load_cases, summary

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "retrieval.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _bm25() -> dict[str, Any]:
    base = load_knowledge_base(ARCHIVE)
    thresholds = SanitizerThresholds()
    screening = screen(base, default_detectors(thresholds), thresholds)
    retriever = LexicalRetriever(admitted_provisions(base, screening))
    return summary(
        evaluate(load_cases(), retriever.retrieve, base, build_catalog(base))
    )


def test_every_source_of_the_golden_set_is_resolved() -> None:
    assert _bm25()["cases"]["unresolved_sources"] == {}


@pytest.mark.parametrize("section", list(EXPECTED["bm25"]))
def test_bm25_numbers_have_not_changed(section: str) -> None:
    assert _bm25()[section] == EXPECTED["bm25"][section]


def test_golden_file_covers_the_whole_summary() -> None:
    assert set(EXPECTED["bm25"]) == set(_bm25())
