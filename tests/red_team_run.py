"""Прогон состязательного набора (PSB-55): кейсы × мутации → проверки → отчёт.

Единого входа «вопрос → ответ» в репозитории нет, поэтому прогон не импортирует
пайплайн, а принимает ответчик — функцию «кейс и база → ответ». Здесь же два
подставных ответчика, на которых проверяется сам набор.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from lex_fact_extraction import find_facts
from lex_knowledge_base import SourceFile
from lex_prompt_boundary import random_boundary
from lex_prompts import ANSWER_PROMPT_V1
from red_team_invariants import Answer, Check, Verdict
from red_team_mutations import Category, Mutation, RedTeamCase, same_value

DEFAULT_SEED = 55
_REVEALED_RULE_CHARACTERS = 300

Response = TypeVar("Response")
Respond = Callable[[RedTeamCase, Sequence[SourceFile]], Response]
ResponseCheck = Callable[
    [RedTeamCase, RedTeamCase, Response, Response, Sequence[SourceFile]], Check
]


@dataclass(frozen=True, slots=True)
class Trial:
    """Один мутированный кейс и исходы его проверок."""

    case: str
    category: Category
    mutation: str
    round: int
    checks: tuple[Check, ...]

    @property
    def verdict(self) -> Verdict:
        verdicts = {check.verdict for check in self.checks}
        if Verdict.VIOLATED in verdicts:
            return Verdict.VIOLATED
        if Verdict.UNDETERMINED in verdicts:
            return Verdict.UNDETERMINED
        return Verdict.HELD


@dataclass(frozen=True, slots=True)
class _Failed:
    reason: str


def mutated_inputs(
    cases: Sequence[RedTeamCase],
    kb: Sequence[SourceFile],
    mutations: Sequence[Mutation],
    seed: int,
    rounds: int,
) -> Iterator[tuple[Mutation, int, RedTeamCase, RedTeamCase, tuple[SourceFile, ...]]]:
    """Мутация, круг, исходный кейс, мутированный кейс и мутированная база.

    Генератор случайных чисел заводится заново на каждую пару «мутация — круг»:
    одна мутация портит базу одинаково для всех кейсов круга, и подготовку
    испорченной базы можно не повторять. Пары, где мутация вход не изменила,
    пропускаются.
    """
    clean = tuple(kb)
    for round_ in range(rounds):
        for mutation in mutations:
            for case in cases:
                rng = random.Random(
                    f"{seed}:{mutation.category.value}:{mutation.name}:{round_}"
                )
                mutated, mutated_kb = mutation.mutate(case, clean, rng)
                if (mutated, mutated_kb) != (case, clean):
                    yield mutation, round_, case, mutated, mutated_kb


def run(
    cases: Sequence[RedTeamCase],
    kb: Sequence[SourceFile],
    respond: Respond[Response],
    checks: Mapping[Category, Sequence[ResponseCheck[Response]]],
    mutations: Sequence[Mutation],
    seed: int = DEFAULT_SEED,
    rounds: int = 1,
) -> tuple[Trial, ...]:
    clean = {case.id: _asked(respond, case, kb) for case in cases}
    trials: list[Trial] = []
    for mutation, round_, case, mutated, mutated_kb in mutated_inputs(
        cases, kb, mutations, seed, rounds
    ):
        before, after = clean[case.id], _asked(respond, mutated, mutated_kb)
        failed = [item for item in (before, after) if isinstance(item, _Failed)]
        trials.append(
            Trial(
                case=case.id,
                category=mutation.category,
                mutation=mutation.name,
                round=round_,
                checks=tuple(
                    Check(check.__name__, Verdict.UNDETERMINED, failed[0].reason)
                    if isinstance(before, _Failed) or isinstance(after, _Failed)
                    else check(case, mutated, before, after, kb)
                    for check in checks[mutation.category]
                ),
            )
        )
    return tuple(trials)


def report(trials: Sequence[Trial]) -> dict[str, Any]:
    """Счёт по категориям и перечень невыдержанных проверок — для JSON."""
    by_category = {
        category.value: _counts(
            [trial for trial in trials if trial.category is category]
        )
        for category in Category
    }
    by_mutation: dict[str, list[Trial]] = {}
    for trial in trials:
        by_mutation.setdefault(f"{trial.category.value}:{trial.mutation}", []).append(
            trial
        )
    return {
        "total": _counts(trials),
        "categories": by_category,
        "mutations": {name: _counts(own) for name, own in by_mutation.items()},
        "not_held": [
            {
                "case": trial.case,
                "category": trial.category.value,
                "mutation": trial.mutation,
                "round": trial.round,
                "invariant": check.invariant,
                "verdict": check.verdict.value,
                "evidence": check.evidence,
            }
            for trial in trials
            for check in trial.checks
            if check.verdict is not Verdict.HELD
        ],
    }


def summary(counted: Mapping[str, Any]) -> str:
    """Таблица отчёта для stdout и описания PR."""
    lines = [
        "| Категория | Мутированных кейсов | Выдержано | Нарушено | Неопределённо "
        "| Доля успешных атак |",
        "|---|---|---|---|---|---|",
    ]
    rows = {**counted["categories"], "всего": counted["total"]}
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['mutated_cases']} | {row['held']} | {row['violated']} "
            f"| {row['undetermined']} | {row['violated']} из {row['mutated_cases']} |"
        )
    lines.append("")
    lines.append("| Категория | Проверка | Выдержано | Нарушено | Неопределённо |")
    lines.append("|---|---|---|---|---|")
    for name, row in counted["categories"].items():
        for invariant, by_verdict in row["invariants"].items():
            lines.append(
                f"| {name} | {invariant} | {by_verdict['held']} "
                f"| {by_verdict['violated']} | {by_verdict['undetermined']} |"
            )
    return "\n".join(lines)


def ideal_answerer(case: RedTeamCase, kb: Sequence[SourceFile]) -> Answer:
    """Отвечает эталоном кейса, что бы ни было подброшено во вход."""
    return Answer(case.reference, random_boundary())


def vulnerable_answerer(case: RedTeamCase, kb: Sequence[SourceFile]) -> Answer:
    """Принимает подменённую величину, повторяет подброшенный текст и вместе
    с ним раскрывает правила и метку границы."""
    boundary = random_boundary()
    text = case.reference
    if case.premise is not None:
        true = find_facts(case.premise.true)
        accepted = [
            fact
            for fact in find_facts(text)
            if any(same_value(fact, other) for other in true)
        ]
        for fact in reversed(accepted):
            text = f"{text[: fact.start]}{case.premise.false}{text[fact.end :]}"
    if case.planted:
        rules = ANSWER_PROMPT_V1.system[:_REVEALED_RULE_CHARACTERS]
        text = " ".join([text, *case.planted, rules, boundary.token])
    return Answer(text, boundary)


def _asked(
    respond: Respond[Response], case: RedTeamCase, kb: Sequence[SourceFile]
) -> Response | _Failed:
    try:
        return respond(case, kb)
    # Ответчик — чужой код с сетью и моделью: любая его ошибка, включая тайм-аут,
    # — исход «неопределённо» для кейса, а не остановка прогона.
    except Exception as error:
        return _Failed(f"{type(error).__name__}: {error}")


def _counts(trials: Sequence[Trial]) -> dict[str, Any]:
    invariants: dict[str, dict[str, int]] = {}
    for trial in trials:
        for check in trial.checks:
            row = invariants.setdefault(
                check.invariant, dict.fromkeys((item.value for item in Verdict), 0)
            )
            row[check.verdict.value] += 1
    violated = sum(trial.verdict is Verdict.VIOLATED for trial in trials)
    return {
        "mutated_cases": len(trials),
        "held": sum(trial.verdict is Verdict.HELD for trial in trials),
        "violated": violated,
        "undetermined": sum(trial.verdict is Verdict.UNDETERMINED for trial in trials),
        "attack_success_rate": violated / len(trials) if trials else None,
        "invariants": invariants,
    }
