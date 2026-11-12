"""Промпт против настоящей базы: в нём нет её фактов, запрос по демо-вопросу цел.

Архив базы в репозиторий не входит: без него тесты пропускаются.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pytest

from demo_answers import ARCHIVE, CASES, base, context, model_reply, questions
from lex_answer_contract import ModelAnswer, generate
from lex_answer_rendering import render
from lex_documents import ProvisionId
from lex_entity_catalog import build_catalog
from lex_entity_recognition import recognize
from lex_models import ChatTurn, FinishReason, JsonCompletion
from lex_normalization import comparison_key
from lex_prompt_boundary import Boundary, assemble_prompt, boundary_violations
from lex_prompts import ANSWER_EXAMPLES_V1, ANSWER_PROMPT_V1, ANSWER_REMINDER_V1
from prompt_conformance import rule_breaches

BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
# «21 день», «1 800 рублей», «12 дней»
NUMBER_WITH_UNIT = re.compile(r"\d[\d\s]*\s[а-яё]+")

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


class _RecordingModel:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.requests: list[list[ChatTurn]] = []

    def complete_json(self, turns: Sequence[ChatTurn]) -> JsonCompletion:
        self.requests.append(list(turns))
        return JsonCompletion(content=self._reply, finish=FinishReason.STOP)


def _base_texts() -> list[str]:
    return [
        comparison_key(provision.text)
        for document in base().documents
        for provision in document.provisions
    ]


def _example_texts() -> list[str]:
    return [
        text
        for example in ANSWER_EXAMPLES_V1
        for text in (
            *(provision.text for provision in example.provisions),
            *(claim.text for claim in example.answer.claims),
        )
    ]


def test_number_with_unit_pattern_reads_facts_of_examples() -> None:
    found = NUMBER_WITH_UNIT.findall("Взнос — 1 800 рублей, срок — 21 день.")

    assert found == ["1 800 рублей", "21 день"]


def test_no_counterparty_or_document_of_the_base_is_named_in_prompt() -> None:
    catalog = build_catalog(base())
    lines = [*ANSWER_PROMPT_V1.system.split("\n"), ANSWER_PROMPT_V1.reminder]

    assert [line for line in lines if recognize(line, catalog)] == []


def test_example_identifiers_are_absent_from_the_base() -> None:
    ids = [p.id for example in ANSWER_EXAMPLES_V1 for p in example.provisions]

    assert [one for one in ids if base().find(ProvisionId(one)) is not None] == []


def test_example_provisions_and_their_numbers_are_absent_from_the_base() -> None:
    texts = _base_texts()
    provisions = [
        comparison_key(provision.text)
        for example in ANSWER_EXAMPLES_V1
        for provision in example.provisions
    ]
    numbers = {
        comparison_key(fact)
        for text in _example_texts()
        for fact in NUMBER_WITH_UNIT.findall(text)
    }

    assert numbers
    assert [one for one in provisions if any(one in text for text in texts)] == []
    assert [
        fact
        for fact in sorted(numbers)
        if any(re.search(rf"(?<!\d){re.escape(fact)}(?!\w)", text) for text in texts)
    ] == []


@pytest.mark.parametrize("case", CASES, ids=[f"demo-{case['demo']}" for case in CASES])
def test_demo_question_goes_through_rules_to_answer_in_reference_form(
    case: dict[str, Any],
) -> None:
    question = questions()[case["demo"]]
    reply, ids = model_reply(case)
    # У вопроса без ответа в базе контекст всё равно не пуст: поиск что-то вернёт.
    provisions = context(
        ids or [base().documents[0].provision_id(base().documents[0].provisions[0])]
    )
    model = _RecordingModel(reply)
    turns = assemble_prompt(ANSWER_PROMPT_V1, provisions, (), question, BOUNDARY)

    generation = generate(model, turns, provisions, BOUNDARY)

    (request,) = model.requests
    system, user = request
    assert system == ChatTurn("system", ANSWER_PROMPT_V1.system)
    assert boundary_violations(request, ANSWER_PROMPT_V1, BOUNDARY) == ()
    assert [one for one in ids if f'id="{one}"' not in user.content] == []
    assert question in user.content
    assert user.content.endswith(ANSWER_REMINDER_V1)
    # Ответ, по которому сверяется форма, написан по правилам промпта.
    assert rule_breaches(ModelAnswer.model_validate_json(reply)) == []
    assert generation.problems == ()
    assert render(generation.answer, base()) == case["rendered"]
