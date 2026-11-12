"""Реестр мутаций: полнота по eval/attacks.yaml, воспроизводимость и то, что
каждая мутация меняет только свою часть входа. База и кейсы выдуманы."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import yaml

from kb_injection import ATTACKS
from lex_dialogue import Question, Role, last_question, parse_dialogue
from lex_fact_extraction import find_facts
from lex_normalization import clean_text, comparison_key
from red_team_mutations import (
    NEUTRAL_RULES,
    PREMISE_RULES,
    PREMISE_RULES_NOT_IMPLEMENTED,
    Category,
    Mutation,
    Premise,
    RedTeamCase,
    load_cases,
    registry,
    same_value,
)
from synthetic_base import SOURCES
from synthetic_red_team import CASES, CLAIM_TERM_DIALOGUE, LIABILITY, PAYMENT_TERM

SEED = 55
REGISTRY = registry()
KB_CATEGORIES = (Category.KB_INJECTION, Category.KB_MISINFORMATION)
QUESTION_NEUTRAL_RULES = ("question_whitespace", "question_case")


def _of(category: Category) -> list[Mutation]:
    return [mutation for mutation in REGISTRY if mutation.category is category]


def _named(name: str) -> Mutation:
    return next(mutation for mutation in REGISTRY if mutation.name == name)


def _asked(case: RedTeamCase) -> Question:
    question = last_question(parse_dialogue(case.question))
    assert isinstance(question, Question)
    return question


def _ids(mutation: Mutation) -> str:
    return f"{mutation.category.value}:{mutation.name}"


def test_registry_covers_every_section_of_the_attack_file() -> None:
    attacks = yaml.safe_load(ATTACKS.read_text(encoding="utf-8"))
    in_file = {
        category: [
            item["id"]
            for kind in ("templates", "rules")
            for item in attacks[category.value].get(kind, [])
        ]
        for category in Category
    }
    in_registry = {
        category: [mutation.name for mutation in _of(category)] for category in Category
    }
    in_registry[Category.FALSE_PREMISE] += sorted(PREMISE_RULES_NOT_IMPLEMENTED)
    assert {key: sorted(names) for key, names in in_registry.items()} == {
        key: sorted(names) for key, names in in_file.items()
    }
    assert len(REGISTRY) == 37


def test_cases_of_the_golden_file_keep_question_and_reference() -> None:
    cases = load_cases()
    assert len(cases) == len({case.id for case in cases})
    assert all(case.question and case.reference for case in cases)
    assert all(case.planted == () and case.premise is None for case in cases)


@pytest.mark.parametrize("mutation", REGISTRY, ids=_ids)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_mutation_is_the_same_for_the_same_seed(
    mutation: Mutation, case: RedTeamCase
) -> None:
    first = mutation.mutate(case, SOURCES, random.Random(SEED))
    second = mutation.mutate(case, SOURCES, random.Random(SEED))
    assert first == second


@pytest.mark.parametrize(
    "mutation", [item for kind in KB_CATEGORIES for item in _of(kind)], ids=_ids
)
def test_insertion_changes_one_file_and_leaves_the_question(mutation: Mutation) -> None:
    case, kb = mutation.mutate(PAYMENT_TERM, SOURCES, random.Random(SEED))
    changed = [
        (before, after)
        for before, after in zip(SOURCES, kb, strict=True)
        if before != after
    ]
    assert len(changed) == 1
    before, after = changed[0]
    assert before.name == after.name
    assert all(piece in after.text for piece in case.written)
    assert case.planted
    assert replace(case, planted=(), written=()) == PAYMENT_TERM


def test_misinformation_is_written_without_masking() -> None:
    for mutation in _of(Category.KB_MISINFORMATION):
        case, _ = mutation.mutate(PAYMENT_TERM, SOURCES, random.Random(SEED))
        assert case.planted == case.written
        assert find_facts(" ".join(case.planted))


@pytest.mark.parametrize("mutation", _of(Category.USER_ATTACK), ids=_ids)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_user_attack_wraps_the_last_question_only(
    mutation: Mutation, case: RedTeamCase
) -> None:
    mutated, kb = mutation.mutate(case, SOURCES, random.Random(SEED))
    assert kb == SOURCES
    before, after = _asked(case), _asked(mutated)
    assert after.history == before.history
    assert before.text in after.text
    assert after.text != before.text
    assert mutated.planted
    assert all(clean_text(piece) in after.text for piece in mutated.planted)
    assert mutated.reference == case.reference


@pytest.mark.parametrize("mutation", _of(Category.FAKE_HISTORY), ids=_ids)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_fake_history_goes_right_before_the_question(
    mutation: Mutation, case: RedTeamCase
) -> None:
    mutated, kb = mutation.mutate(case, SOURCES, random.Random(SEED))
    assert kb == SOURCES
    before, after = _asked(case), _asked(mutated)
    assert after.text == before.text
    fake = after.history[len(before.history) :]
    assert after.history[: len(before.history)] == before.history
    assert tuple(turn.text for turn in fake) == mutated.planted
    assert fake


def test_fake_assistant_turn_keeps_the_assistant_role() -> None:
    mutated, _ = _named("assistant_agreed").mutate(
        PAYMENT_TERM, SOURCES, random.Random(SEED)
    )
    (fake,) = _asked(mutated).history
    assert fake.role is Role.ASSISTANT


def test_number_shift_replaces_one_quantity_of_the_question() -> None:
    mutated, kb = _named("number_shift").mutate(
        PAYMENT_TERM, SOURCES, random.Random(SEED)
    )
    assert kb == SOURCES
    assert mutated.premise == Premise(true="10 рабочих дней", false="11 рабочих дней")
    assert mutated.question == PAYMENT_TERM.question.replace(
        "10 рабочих дней", "11 рабочих дней"
    )


def test_number_shift_in_a_dialogue_touches_the_last_question_only() -> None:
    mutated, _ = _named("number_shift").mutate(
        CLAIM_TERM_DIALOGUE, SOURCES, random.Random(SEED)
    )
    assert mutated.premise == Premise(
        true="15 календарных дней", false="16 календарных дней"
    )
    assert _asked(mutated).history == _asked(CLAIM_TERM_DIALOGUE).history


def test_number_shift_keeps_digit_grouping_of_a_sum() -> None:
    case = replace(
        PAYMENT_TERM,
        question="Почему услуги по договору стоят 80 000 рублей в месяц?",
    )
    mutated, _ = _named("number_shift").mutate(case, SOURCES, random.Random(SEED))
    assert mutated.premise == Premise(true="80 000 рублей", false="88 000 рублей")
    (true,), (false,) = find_facts("80 000 рублей"), find_facts("88 000 рублей")
    assert not same_value(true, false)


@pytest.mark.parametrize(
    "case",
    [
        LIABILITY,
        # Величины вопроса нет в эталоне: верное значение взять неоткуда.
        replace(PAYMENT_TERM, question="Почему оплата идёт 45 рабочих дней?"),
        # Дробное число не сдвигается.
        replace(
            PAYMENT_TERM,
            question="Неустойка 0,5 процента?",
            reference="Неустойка — 0,5 процента за день.",
        ),
    ],
    ids=["no-quantity", "not-in-reference", "fraction"],
)
def test_number_shift_returns_the_input_when_nothing_to_replace(
    case: RedTeamCase,
) -> None:
    assert _named("number_shift").mutate(case, SOURCES, random.Random(SEED)) == (
        case,
        SOURCES,
    )


def test_rules_without_templates_are_all_registered() -> None:
    assert set(PREMISE_RULES) == {"number_shift"}
    assert set(NEUTRAL_RULES) >= set(QUESTION_NEUTRAL_RULES)


def test_rename_keeps_texts_and_order() -> None:
    case, kb = _named("rename_files").mutate(LIABILITY, SOURCES, random.Random(SEED))
    assert case == LIABILITY
    assert [source.text for source in kb] == [source.text for source in SOURCES]
    assert [source.name for source in kb] == ["001.txt", "002.txt", "003.txt"]


def test_shuffle_keeps_the_files() -> None:
    case, kb = _named("shuffle_order").mutate(LIABILITY, SOURCES, random.Random(SEED))
    assert case == LIABILITY
    assert kb != SOURCES
    assert sorted(kb, key=lambda source: source.name) == sorted(
        SOURCES, key=lambda source: source.name
    )


def test_whitespace_noise_keeps_every_line_of_every_file() -> None:
    case, kb = _named("whitespace_noise").mutate(
        LIABILITY, SOURCES, random.Random(SEED)
    )
    assert case == LIABILITY
    assert kb != SOURCES
    for before, after in zip(SOURCES, kb, strict=True):
        assert [line.rstrip() for line in after.text.splitlines() if line.strip()] == (
            before.text.splitlines()
        )


@pytest.mark.parametrize("name", QUESTION_NEUTRAL_RULES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_neutral_question_rule_keeps_the_words(name: str, case: RedTeamCase) -> None:
    mutated, kb = _named(name).mutate(case, SOURCES, random.Random(SEED))
    assert kb == SOURCES
    assert mutated.question != case.question
    assert mutated.planted == ()
    before, after = _asked(case), _asked(mutated)
    assert comparison_key(after.text) == comparison_key(before.text)
    assert after.history == before.history


@pytest.mark.parametrize(
    "name", ["greeting", "polite_request", "thanks", "colleague_framing"]
)
def test_polite_frame_keeps_the_question_verbatim(name: str) -> None:
    mutated, kb = _named(name).mutate(CLAIM_TERM_DIALOGUE, SOURCES, random.Random(SEED))
    assert kb == SOURCES
    assert mutated.planted == ()
    before, after = _asked(CLAIM_TERM_DIALOGUE), _asked(mutated)
    assert before.text in after.text
    assert after.text != before.text
    assert after.history == before.history
