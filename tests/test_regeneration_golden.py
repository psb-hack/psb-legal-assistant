"""Генерация с проверкой на настоящей базе: демо-ответы и их искажения.

Модель подставная: отвечает вручную составленным ответом или его искажённым
вариантом. Архив базы в репозиторий не входит: без него тесты пропускаются.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from demo_answers import (
    ARCHIVE,
    CASES,
    admitted_context,
    base,
    graph,
    model_reply,
    questions,
    screening,
)
from lex_answer_rendering import render
from lex_prompt_boundary import (
    Boundary,
    PromptRules,
    assemble_prompt,
    boundary_violations,
)
from lex_prompts import ANSWER_PROMPT_V1
from lex_regeneration import AnswerPath, VerifiedAnswer, generate_verified
from scripted_model import ScriptedJsonModel

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN: dict[str, Any] = yaml.safe_load(
    (_GOLDEN_DIR / "regeneration.yaml").read_text(encoding="utf-8")
)
DISTORTIONS: list[dict[str, Any]] = yaml.safe_load(
    (_GOLDEN_DIR / "verification.yaml").read_text(encoding="utf-8")
)["distortions"]
BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
CASES_BY_DEMO = {case["demo"]: case for case in CASES}

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


def _distorted(distortion: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Ответ с одной правкой текста утверждения и сама искажённая фраза."""
    case = copy.deepcopy(CASES_BY_DEMO[distortion["demo"]])
    claim = case["claims"][distortion["claim"]]
    assert distortion["replace"] in claim["text"]
    claim["text"] = claim["text"].replace(distortion["replace"], distortion["with"])
    return case, claim["text"].rstrip(".")


def _run(demo: str, *cases: dict[str, Any]) -> tuple[VerifiedAnswer, ScriptedJsonModel]:
    replies = [model_reply(case) for case in cases]
    context = admitted_context(replies[0][1])
    question = questions()[demo]
    turns = assemble_prompt(ANSWER_PROMPT_V1, context, (), question, BOUNDARY)
    model = ScriptedJsonModel(*(reply for reply, _ in replies))
    result = generate_verified(
        model, turns, context, BOUNDARY, question, screening(), graph()
    )
    assert len(model.requests) == result.model_calls
    return result, model


def test_correct_demo_answers_are_accepted_with_one_call() -> None:
    paths: Counter[str] = Counter()
    for demo, case in CASES_BY_DEMO.items():
        result, _ = _run(demo, case)
        paths[result.path.value] += 1
        assert result.model_calls == 1
        assert render(result.answer, base()) == case["rendered"]
    assert dict(paths) == GOLDEN["correct"]


def test_distorted_answer_followed_by_correct_one_gives_correct_text() -> None:
    paths: Counter[str] = Counter()
    calls = 0
    for distortion in DISTORTIONS:
        case = CASES_BY_DEMO[distortion["demo"]]
        wrong, wrong_claim = _distorted(distortion)
        result, model = _run(distortion["demo"], wrong, case)
        paths[result.path.value] += 1
        calls += result.model_calls
        text = render(result.answer, base())
        assert wrong_claim not in text
        if result.second is not None:
            assert text == case["rendered"]
            first, second = model.requests
            addition = second[1].content[len(first[1].content) :]
            extended = PromptRules(
                ANSWER_PROMPT_V1.system, ANSWER_PROMPT_V1.reminder + addition
            )
            assert boundary_violations(list(second), extended, BOUNDARY) == ()
            assert wrong_claim not in addition
            assert distortion["with"] not in addition
    assert dict(paths) == GOLDEN["distorted_then_correct"]["paths"]
    assert calls == GOLDEN["distorted_then_correct"]["model_calls"]


def test_distortion_repeated_by_second_answer_never_reaches_text() -> None:
    paths: Counter[str] = Counter()
    unconfirmed: dict[str, str] = {}
    for distortion in DISTORTIONS:
        wrong, wrong_claim = _distorted(distortion)
        result, _ = _run(distortion["demo"], wrong, wrong)
        paths[result.path.value] += 1
        text = render(result.answer, base())
        assert wrong_claim not in text
        assert result.model_calls <= 2
        if result.path is AnswerPath.UNCONFIRMED:
            unconfirmed[distortion["name"]] = text
    assert dict(paths) == GOLDEN["distorted_twice"]["paths"]
    assert unconfirmed == GOLDEN["distorted_twice"]["unconfirmed"]
