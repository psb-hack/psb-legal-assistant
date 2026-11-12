"""Сборка контекста на настоящей базе. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from lex_documents import cite
from lex_knowledge_base import load_knowledge_base
from lex_normalization import detection_alignment, normalize
from lex_prompt_boundary import (
    Boundary,
    PromptRules,
    assemble_prompt,
    boundary_violations,
)
from lex_role_markers import neutralize_role_markers
from lex_sanitizer import AdmittedProvision, admit, default_detectors, screen
from lex_settings import SanitizerThresholds

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "prompt_boundary.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
RULES = PromptRules(
    system="Отвечай только по блокам provision.", reminder="Ответ — JSON."
)

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _admitted() -> tuple[tuple[str, AdmittedProvision], ...]:
    """Ссылка на положение и положение, допущенное санитайзером в контекст."""
    thresholds = SanitizerThresholds()
    base = load_knowledge_base(ARCHIVE)
    screening = screen(base, default_detectors(thresholds), thresholds)
    return tuple(
        (cite(document, provision), admitted)
        for document in base.documents
        for provision in document.provisions
        if (admitted := admit(document, provision, screening)) is not None
    )


def test_neutralization_changes_no_provision_of_the_base() -> None:
    admitted = _admitted()
    changed = [
        citation
        for citation, provision in admitted
        if neutralize_role_markers(provision.text) != provision.text
        or neutralize_role_markers(provision.heading) != provision.heading
    ]
    assert len(admitted) == EXPECTED["admitted_provisions"]
    assert changed == EXPECTED["changed"], f"{len(changed)} из {len(admitted)}"


def test_whole_base_as_context_passes_the_boundary_check() -> None:
    provisions = tuple(provision for _, provision in _admitted())
    turns = assemble_prompt(RULES, provisions, (), "Какой срок оплаты?", BOUNDARY)
    assert boundary_violations(turns, RULES, BOUNDARY) == ()
    for provision in provisions:
        assert provision.text in turns[1].content


def test_aligned_key_matches_the_detection_key_on_every_provision() -> None:
    different = [
        citation
        for citation, provision in _admitted()
        if " ".join(detection_alignment(provision.text).key.split())
        != normalize(provision.text).detection_key
    ]
    assert different == EXPECTED["aligned_key_differs"]
