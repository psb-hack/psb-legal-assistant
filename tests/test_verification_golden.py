"""Верификатор на настоящей базе: верные демо-ответы и их искажённые варианты.

Архив базы в репозиторий не входит: без него тесты пропускаются.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from demo_answers import ARCHIVE, CASES, base, model_reply, questions
from lex_answer_contract import GroundedAnswer, GroundedClaim, read_reply
from lex_document_graph import DocumentGraph, build_graph
from lex_documents import ProvisionId
from lex_models import FinishReason, JsonCompletion
from lex_prompt_boundary import Boundary
from lex_sanitizer import (
    AdmittedProvision,
    Screening,
    admit,
    default_detectors,
    screen,
)
from lex_settings import SanitizerThresholds
from lex_verification import VerificationReport, ViolationKind, verify

GOLDEN: dict[str, Any] = yaml.safe_load(
    (Path(__file__).resolve().parent / "golden" / "verification.yaml").read_text(
        encoding="utf-8"
    )
)
BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
CASES_BY_DEMO = {case["demo"]: case for case in CASES}

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _screening() -> Screening:
    thresholds = SanitizerThresholds()
    return screen(base(), default_detectors(thresholds), thresholds)


@functools.cache
def _graph() -> DocumentGraph:
    return build_graph(base())


def _context(ids: Sequence[ProvisionId]) -> list[AdmittedProvision]:
    """Допущенные положения документов, на которые ссылается ответ."""
    cited = {one.split(".")[0] for one in ids}
    admitted = (
        admit(document, provision, _screening())
        for document in base().documents
        if document.id in cited
        for provision in document.provisions
    )
    return [provision for provision in admitted if provision is not None]


def _demo(demo: str) -> tuple[GroundedAnswer, list[AdmittedProvision]]:
    reply, ids = model_reply(CASES_BY_DEMO[demo])
    context = _context(ids)
    reading = read_reply(JsonCompletion(reply, FinishReason.STOP), context, BOUNDARY)
    assert reading.grounded is not None
    assert reading.problems == ()
    return reading.grounded, context


def _verify(
    demo: str, answer: GroundedAnswer, context: Sequence[AdmittedProvision]
) -> VerificationReport:
    return verify(answer, context, questions()[demo], _screening(), _graph())


def test_correct_demo_answers_have_no_rejected_claims() -> None:
    confirmed = total = 0
    support: Counter[str] = Counter()
    for demo in CASES_BY_DEMO:
        answer, context = _demo(demo)
        report = _verify(demo, answer, context)
        assert not any(item.violations for item in report.missing)
        total += len(report.claims)
        confirmed += sum(claim.confirmed for claim in report.claims)
        support.update(
            fact.support.value for claim in report.claims for fact in claim.facts
        )
    assert {"confirmed": confirmed, "total": total} == GOLDEN["claims"]
    assert dict(support) == GOLDEN["facts"]


@pytest.mark.parametrize(
    "distortion", GOLDEN["distortions"], ids=lambda item: str(item["name"])
)
def test_distorted_demo_answer_is_caught(distortion: dict[str, Any]) -> None:
    answer, context = _demo(distortion["demo"])
    index = distortion["claim"]
    original = answer.claims[index]
    assert distortion["replace"] in original.text
    distorted = GroundedClaim(
        original.provisions,
        original.quote,
        original.text.replace(distortion["replace"], distortion["with"]),
    )
    claims = (*answer.claims[:index], distorted, *answer.claims[index + 1 :])
    changed = GroundedAnswer(answer.scenario, claims, answer.missing, answer.verdict)
    report = _verify(distortion["demo"], changed, context)
    assert report.claims[index].violations == (ViolationKind[distortion["violation"]],)
    others = [check for place, check in enumerate(report.claims) if place != index]
    assert all(check.confirmed for check in others)
