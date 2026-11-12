"""Распознавание упоминаний на настоящей базе и демо-вопросах организаторов."""

from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_entity_catalog import Counterparty, EntityCatalog, build_catalog
from lex_entity_recognition import (
    Ambiguous,
    Mention,
    Resolved,
    combine_turns,
    recognize,
)
from lex_knowledge_base import load_knowledge_base

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
DEMO_QUESTIONS = REPO / "data" / "demo_questions.csv"
GOLDEN = Path(__file__).resolve().parent / "golden" / "entity_recognition.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
# Разбор диалога на реплики — задача PSB-24; тесту хватает метки в начале строки.
USER_LABEL = "Пользователь:"

pytestmark = pytest.mark.skipif(
    not (ARCHIVE.is_file() and DEMO_QUESTIONS.is_file()),
    reason="нет data/knowledge-base.zip или data/demo_questions.csv",
)


@functools.cache
def _catalog() -> EntityCatalog:
    return build_catalog(load_knowledge_base(ARCHIVE))


@functools.cache
def _demo_questions() -> dict[str, str]:
    with DEMO_QUESTIONS.open(encoding="utf-8-sig", newline="") as file:
        return {row["id"]: row["question"] for row in csv.DictReader(file)}


def _summary(mention: Mention) -> dict[str, Any]:
    resolution = mention.resolution
    if isinstance(resolution, Resolved):
        outcome, entities = "resolved", (resolution.entity,)
        evidence = resolution.evidence.name
    elif isinstance(resolution, Ambiguous):
        outcome, entities = "ambiguous", resolution.candidates
        evidence = resolution.evidence.name
    else:
        outcome, entities, evidence = "not_found", resolution.similar, None
    return {
        "text": mention.text,
        "outcome": outcome,
        "entities": [
            entity.name if isinstance(entity, Counterparty) else entity.title
            for entity in entities
        ],
        "evidence": evidence,
    }


def test_golden_covers_every_demo_question() -> None:
    assert set(EXPECTED["demo_questions"]) == set(_demo_questions())


@pytest.mark.parametrize("question_id", list(EXPECTED["demo_questions"]))
def test_mentions_in_demo_question(question_id: str) -> None:
    mentions = recognize(_demo_questions()[question_id], _catalog())
    assert [_summary(mention) for mention in mentions] == (
        EXPECTED["demo_questions"][question_id]
    )


@pytest.mark.parametrize("question_id", list(EXPECTED["user_turns"]))
def test_user_turns_are_recognized_separately_and_combined(question_id: str) -> None:
    turns = [
        recognize(line, _catalog())
        for line in _demo_questions()[question_id].splitlines()
        if line.startswith(USER_LABEL)
    ]
    assert [_summary(mention) for mention in combine_turns(turns)] == (
        EXPECTED["user_turns"][question_id]
    )


@pytest.mark.parametrize("case", EXPECTED["phrases"], ids=lambda case: case["text"])
def test_mentions_in_phrase(case: dict[str, Any]) -> None:
    mentions = recognize(case["text"], _catalog())
    assert [_summary(mention) for mention in mentions] == case["mentions"]
