"""Форма ответа на настоящей базе против эталонов организаторов.

Архив базы в репозиторий не входит: без него тесты пропускаются.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from demo_answers import ARCHIVE, CASES, base, context, model_reply, reference_answers
from lex_answer_contract import generate
from lex_answer_rendering import render
from lex_models import ChatTurn, FinishReason, JsonCompletion
from lex_normalization import comparison_key
from lex_prompt_boundary import Boundary

BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
TURNS = [ChatTurn("system", "Правила."), ChatTurn("user", "Вопрос.")]

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


class _OneReplyModel:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete_json(self, turns: Sequence[ChatTurn]) -> JsonCompletion:
        return JsonCompletion(content=self._reply, finish=FinishReason.STOP)


@pytest.mark.parametrize("case", CASES, ids=[f"demo-{case['demo']}" for case in CASES])
def test_structured_answer_renders_in_the_form_of_reference_answer(
    case: dict[str, Any],
) -> None:
    reply, ids = model_reply(case)
    generation = generate(_OneReplyModel(reply), TURNS, context(ids), BOUNDARY)
    assert generation.problems == ()
    assert generation.extra_call_spent is False

    text = render(generation.answer, base())

    assert text == case["rendered"]
    reference = comparison_key(reference_answers()[case["demo"]])
    for fragment in case["shared"]:
        assert comparison_key(fragment) in comparison_key(text), fragment
        assert comparison_key(fragment) in reference, fragment
