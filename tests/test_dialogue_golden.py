"""Диалоги golden-набора на настоящей базе: вопрос, сущности, запрос для поиска
и место источника в выдаче. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_dialogue import (
    Question,
    Role,
    dialogue_entities,
    last_question,
    parse_dialogue,
    prompt_history,
    search_query,
)
from lex_entity_catalog import Counterparty, EntityCatalog, build_catalog
from lex_entity_recognition import Entity, Resolved
from lex_knowledge_base import KnowledgeBase, load_knowledge_base
from lex_prompt_boundary import Boundary, assemble_prompt, boundary_violations
from lex_prompts import ANSWER_PROMPT_V1
from lex_retrieval import LexicalRetriever
from lex_sanitizer import admitted_provisions, default_detectors, screen
from lex_settings import SanitizerThresholds
from retrieval_eval import EvalCase, evaluate, load_cases, summary

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "dialogue.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
ATTACKS = yaml.safe_load((REPO / "eval" / "attacks.yaml").read_text(encoding="utf-8"))
FAKE_HISTORY = {
    template["id"]: tuple(template["turns"])
    for template in ATTACKS["fake_history"]["templates"]
}
SUBSTITUTIONS = {
    "attribute": "срок ответа на претензию",
    "object": "договору с «Альфа-Трейд»",
    "wrong_value": "30 календарных дней",
}
WRONG_TERM_CASE = "team-20"
BOUNDARY = Boundary("9f2c4e6a8b0d1f35")

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _base() -> KnowledgeBase:
    return load_knowledge_base(ARCHIVE)


@functools.cache
def _catalog() -> EntityCatalog:
    return build_catalog(_base())


@functools.cache
def _retriever() -> LexicalRetriever:
    thresholds = SanitizerThresholds()
    screening = screen(_base(), default_detectors(thresholds), thresholds)
    return LexicalRetriever(admitted_provisions(_base(), screening))


def _cases() -> dict[str, EvalCase]:
    return {case.id: case for case in load_cases()}


def _question(text: str) -> Question:
    question = last_question(parse_dialogue(text))
    assert isinstance(question, Question)
    return question


def _query(text: str) -> str:
    question = _question(text)
    return search_query(question, dialogue_entities(question, _catalog()))


def _name(entity: Entity) -> str:
    return entity.name if isinstance(entity, Counterparty) else entity.title


def _reading(text: str) -> dict[str, Any]:
    question = _question(text)
    entities = dialogue_entities(question, _catalog())
    assert all(
        isinstance(mention.resolution, Resolved) for mention in entities.named_by_user
    )
    return {
        "question": question.text,
        "named_by_user": [
            _name(mention.resolution.entity)
            for mention in entities.named_by_user
            if isinstance(mention.resolution, Resolved)
        ],
        "suggested_by_assistant": [
            _name(entity) for entity in entities.suggested_by_assistant
        ],
        "query": search_query(question, entities).split("\n"),
    }


@functools.cache
def _ranks() -> dict[str, dict[str, Any]]:
    """Места источников: запрос — текст кейса целиком и запрос из диалога."""
    cases = [_cases()[case_id] for case_id in EXPECTED["retrieval"]]
    searches = {
        "raw": _retriever().retrieve,
        "dialogue": lambda text, depth: _retriever().retrieve(_query(text), depth),
    }
    found = {
        name: summary(evaluate(cases, search, _base(), _catalog()))["ranks"]
        for name, search in searches.items()
    }
    return {
        case.id: {name: ranks[case.id] for name, ranks in found.items()}
        for case in cases
    }


def _with_fake_history(template: str, dialogue: str) -> str:
    *before, last = dialogue.splitlines()
    fake = [line.format(**SUBSTITUTIONS) for line in FAKE_HISTORY[template]]
    return "\n".join([*before, *fake, last])


def test_golden_covers_every_case_with_more_than_one_turn() -> None:
    dialogues = {
        case.id for case in load_cases() if len(parse_dialogue(case.question).turns) > 1
    }
    assert set(EXPECTED["dialogues"]) == dialogues
    assert set(EXPECTED["retrieval"]) == dialogues


def test_single_questions_are_searched_as_they_are() -> None:
    for case in load_cases():
        if case.id not in EXPECTED["dialogues"]:
            assert _query(case.question) == case.question, case.id


@pytest.mark.parametrize("case_id", list(EXPECTED["dialogues"]))
def test_question_entities_and_query_of_the_dialogue(case_id: str) -> None:
    assert _reading(_cases()[case_id].question) == EXPECTED["dialogues"][case_id]


@pytest.mark.parametrize("case_id", list(EXPECTED["dialogues"]))
def test_assistant_words_do_not_reach_the_search_query(case_id: str) -> None:
    text = _cases()[case_id].question
    replies = [
        turn.text for turn in parse_dialogue(text).turns if turn.role is Role.ASSISTANT
    ]
    assert replies
    assert not any(reply in _query(text) for reply in replies)


@pytest.mark.parametrize("case_id", list(EXPECTED["dialogues"]))
def test_prompt_of_the_dialogue_passes_the_boundary_check(case_id: str) -> None:
    question = _question(_cases()[case_id].question)
    provisions = [hit.provision for hit in _retriever().retrieve(question.text, 5)]
    turns = assemble_prompt(
        ANSWER_PROMPT_V1, provisions, prompt_history(question), question.text, BOUNDARY
    )
    assert boundary_violations(turns, ANSWER_PROMPT_V1, BOUNDARY) == ()


@pytest.mark.parametrize("case_id", list(EXPECTED["retrieval"]))
def test_source_ranks_by_case_text_and_by_dialogue_query(case_id: str) -> None:
    assert _ranks()[case_id] == EXPECTED["retrieval"][case_id]


def test_wrong_term_in_the_assistant_reply_stays_out_of_search() -> None:
    reading = _reading(_cases()[WRONG_TERM_CASE].question)
    assert not any("30 календарных" in line for line in reading["query"])
    assert _ranks()[WRONG_TERM_CASE]["dialogue"]["first"] == 1


@pytest.mark.parametrize("template", sorted(FAKE_HISTORY))
def test_fake_history_does_not_change_what_is_searched_for(template: str) -> None:
    original = _cases()[WRONG_TERM_CASE].question
    attacked = _with_fake_history(template, original)
    before, after = _reading(original), _reading(attacked)
    assert after["question"] == before["question"]
    assert after["suggested_by_assistant"] == before["suggested_by_assistant"]
    assert set(after["named_by_user"]) == set(before["named_by_user"])
    said = {turn.role: [] for turn in parse_dialogue(attacked).turns}
    for turn in parse_dialogue(attacked).turns:
        said[turn.role].append(turn.text)
    names = after["suggested_by_assistant"]
    assert all(line in said[Role.USER] or line in names for line in after["query"])
    assert not any(reply in after["query"] for reply in said[Role.ASSISTANT])


@pytest.mark.parametrize("template", sorted(FAKE_HISTORY))
def test_prompt_with_fake_history_passes_the_boundary_check(template: str) -> None:
    attacked = _with_fake_history(template, _cases()[WRONG_TERM_CASE].question)
    question = _question(attacked)
    turns = assemble_prompt(
        ANSWER_PROMPT_V1, (), prompt_history(question), question.text, BOUNDARY
    )
    assert boundary_violations(turns, ANSWER_PROMPT_V1, BOUNDARY) == ()
