"""Санитайзер на настоящей базе. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from kb_injection import recall_table, run_cases
from lex_documents import cite
from lex_knowledge_base import KnowledgeBase, load_knowledge_base, read_archive
from lex_sanitizer import (
    Screening,
    Verdict,
    admitted_text,
    default_detectors,
    screen,
    visible_fragments,
)
from lex_settings import SanitizerThresholds

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "quarantine.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
COMPANY_FOLDER = "документы компании"

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _screened() -> tuple[KnowledgeBase, Screening]:
    thresholds = SanitizerThresholds()
    base = load_knowledge_base(ARCHIVE)
    return base, screen(base, default_detectors(thresholds), thresholds)


def _citations(base: KnowledgeBase) -> dict[str, str]:
    """Ссылка на положение → текст, который санитайзер пропускает в индекс."""
    _, screening = _screened()
    return {
        cite(document, provision): admitted_text(document, provision, screening)
        for document in base.documents
        for provision in document.provisions
    }


def test_only_the_known_insertion_is_quarantined() -> None:
    base, screening = _screened()
    fragments = sum(len(visible_fragments(document)) for document in base.documents)
    assert fragments == EXPECTED["fragments"]
    quarantined = []
    for decision in screening.quarantined:
        assert decision.provision_id is not None
        found = base.find(decision.provision_id)
        assert found is not None
        quarantined.append(
            {"citation": cite(*found), "detector": decision.signals[0].detector}
        )
    extra = len(quarantined) - len(EXPECTED["quarantined"])
    assert quarantined == EXPECTED["quarantined"], f"лишних: {extra} из {fragments}"


def test_no_fragment_of_the_base_waits_for_the_classifier() -> None:
    _, screening = _screened()
    suspects = [item for item in screening.decisions if item.verdict is Verdict.SUSPECT]
    assert len(suspects) == EXPECTED["suspects"]


def test_neighbours_of_the_insertion_stay_in_the_index() -> None:
    base, _ = _screened()
    admitted = _citations(base)
    for item in EXPECTED["quarantined"]:
        assert admitted[item["citation"]] == ""
    for citation in EXPECTED["kept_neighbours"]:
        assert admitted[citation]


def test_recall_on_company_documents_is_as_recorded() -> None:
    sources = [item for item in read_archive(ARCHIVE) if COMPANY_FOLDER in item.name]
    outcomes = run_cases(sources, EXPECTED["recall_seed"])
    measured = {
        template: list(row.values()) for template, row in recall_table(outcomes).items()
    }
    assert measured == EXPECTED["recall"]
    assert [outcome.collateral for outcome in outcomes if outcome.collateral] == []
