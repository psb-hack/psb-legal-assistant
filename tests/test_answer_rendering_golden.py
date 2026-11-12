"""Форма ответа на настоящей базе против эталонов организаторов.

Архив базы в репозиторий не входит: без него тесты пропускаются.
"""

from __future__ import annotations

import csv
import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_answer_contract import generate
from lex_answer_rendering import render
from lex_documents import ProvisionId, cite
from lex_knowledge_base import KnowledgeBase, load_knowledge_base
from lex_models import ChatTurn, FinishReason, JsonCompletion
from lex_normalization import comparison_key
from lex_prompt_boundary import Boundary
from lex_sanitizer import AdmittedProvision

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
REFERENCE_ANSWERS = REPO / "data" / "demo_output.csv"
GOLDEN = Path(__file__).resolve().parent / "golden" / "answer_rendering.yaml"
CASES = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
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


@functools.cache
def _base() -> KnowledgeBase:
    return load_knowledge_base(ARCHIVE)


@functools.cache
def _ids_by_reference() -> dict[str, ProvisionId]:
    return {
        cite(document, provision): document.provision_id(provision)
        for document in _base().documents
        for provision in document.provisions
    }


@functools.cache
def _reference_answers() -> dict[str, str]:
    with REFERENCE_ANSWERS.open(encoding="utf-8-sig", newline="") as file:
        return {row["id"]: row["answer"] for row in csv.DictReader(file)}


def _context(ids: Sequence[ProvisionId]) -> list[AdmittedProvision]:
    """Документы, на которые ссылается ответ, целиком — как их дал бы поиск."""
    cited_documents = {one.split(".")[0] for one in ids}
    return [
        AdmittedProvision(
            id=document.provision_id(provision),
            heading=cite(document, provision),
            text=provision.text,
        )
        for document in _base().documents
        if document.id in cited_documents
        for provision in document.provisions
    ]


def _model_reply(case: dict[str, Any]) -> tuple[str, list[ProvisionId]]:
    ids: list[ProvisionId] = []
    claims = []
    for claim in case["claims"]:
        cited = [_ids_by_reference()[source] for source in claim["sources"]]
        ids.extend(cited)
        claims.append({"ids": cited, "quote": claim["quote"], "text": claim["text"]})
    reply = {
        "scenario": case["scenario"],
        "claims": claims,
        "missing": case["missing"],
        "verdict": case["verdict"],
    }
    return json.dumps(reply, ensure_ascii=False), ids


@pytest.mark.parametrize("case", CASES, ids=[f"demo-{case['demo']}" for case in CASES])
def test_structured_answer_renders_in_the_form_of_reference_answer(
    case: dict[str, Any],
) -> None:
    reply, ids = _model_reply(case)
    generation = generate(_OneReplyModel(reply), TURNS, _context(ids), BOUNDARY)
    assert generation.problems == ()
    assert generation.extra_call_spent is False

    text = render(generation.answer, _base())

    assert text == case["rendered"]
    reference = comparison_key(_reference_answers()[case["demo"]])
    for fragment in case["shared"]:
        assert comparison_key(fragment) in comparison_key(text), fragment
        assert comparison_key(fragment) in reference, fragment
