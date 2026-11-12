"""Реестр мутаций состязательного и метаморфного набора (PSB-55).

Мутация — функция mutate(case, kb, rng) -> (case, kb): портит вопрос кейса или
базу по разделу eval/attacks.yaml и записывает в кейс, что подброшено, — по этой
записи инварианты узнают чужой текст в ответе. Мутация, которой в кейсе нечего
менять, возвращает вход как есть: прогон такие пары пропускает.

Вставки в базу делает генератор kb_injection, здесь он только вызывается.
"""

from __future__ import annotations

import functools
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

import kb_injection
from kb_injection import (
    ATTACKS,
    ATTRIBUTES,
    OBJECTS,
    WRONG_VALUES,
    InjectionCase,
    Obfuscation,
    Position,
)
from lex_dialogue import Role, Turn, parse_dialogue
from lex_fact_extraction import Fact, Quantity, find_facts
from lex_knowledge_base import SourceFile

CASES = Path(__file__).resolve().parents[1] / "eval" / "cases.yaml"

Sources = tuple[SourceFile, ...]

_LINE_BREAK = "\n"
_QUESTION_SLOT = "{question}"
# «500 000», «12», «5» в начале записи величины; «0,5» и «1.5» не подходят:
# сдвиг целой части дал бы число, которое читается как опечатка.
_WHOLE_NUMBER = re.compile(r"\d+(?:\s\d{3})*(?![.,]?\d)")
_DIGIT_GROUP = 3
# «Соседнее значение» из правила number_shift: на десятую часть, но не меньше
# чем на единицу, — срок в днях сдвигается на день, сумма — на 10%.
_SHIFT_DIVISOR = 10
_BLANK_LINE_EVERY = 4
_SPACE_RUNS = (" ", "  ", "   ")


class Category(Enum):
    """Раздел eval/attacks.yaml; по категориям считается доля успешных атак."""

    KB_INJECTION = "kb_injection"
    KB_MISINFORMATION = "kb_misinformation"
    USER_ATTACK = "user_attack"
    FAKE_HISTORY = "fake_history"
    FALSE_PREMISE = "false_premise"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class Premise:
    # Величина вопроса, как она была написана: «10 рабочих дней».
    true: str
    # Чем её заменила мутация: «11 рабочих дней».
    false: str


@dataclass(frozen=True, slots=True)
class RedTeamCase:
    id: str
    scenario: str
    # Одиночный вопрос или диалог «Пользователь: … / Ассистент: …».
    question: str
    # Эталонный ответ кейса: из него собирает ответ подставной ответчик.
    reference: str
    # Чужой текст, который мутация добавила в базу или в диалог, без маскировки.
    planted: tuple[str, ...] = ()
    # Он же в том виде, в каком записан во вход (с маскировкой, если она была).
    written: tuple[str, ...] = ()
    premise: Premise | None = None


Mutate = Callable[
    [RedTeamCase, Sequence[SourceFile], random.Random], tuple[RedTeamCase, Sources]
]


@dataclass(frozen=True, slots=True)
class Mutation:
    category: Category
    # Идентификатор шаблона или правила в eval/attacks.yaml.
    name: str
    mutate: Mutate


def load_cases(path: Path = CASES) -> tuple[RedTeamCase, ...]:
    return tuple(
        RedTeamCase(
            id=item["id"],
            scenario=item["scenario"],
            question=item["question"].strip(),
            reference=item["reference"].strip(),
        )
        for item in yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def registry() -> tuple[Mutation, ...]:
    """Все мутации набора в порядке разделов eval/attacks.yaml."""
    attacks = yaml.safe_load(ATTACKS.read_text(encoding="utf-8"))
    return (
        *(
            Mutation(
                Category.KB_INJECTION, name, _kb_insertion(name, parts, Obfuscation)
            )
            for name, parts in kb_injection.load_templates().items()
        ),
        *(
            Mutation(
                Category.KB_MISINFORMATION,
                item["id"],
                # Дезинформация не прячется: это обычный на вид абзац документа.
                _kb_insertion(item["id"], (item["text"],), (Obfuscation.NONE,)),
            )
            for item in _templates(attacks, Category.KB_MISINFORMATION)
        ),
        *(
            Mutation(
                Category.USER_ATTACK,
                item["id"],
                _question_frame(item["text"], is_attack=True),
            )
            for item in _templates(attacks, Category.USER_ATTACK)
        ),
        *(
            Mutation(
                Category.FAKE_HISTORY, item["id"], _fake_history(tuple(item["turns"]))
            )
            for item in _templates(attacks, Category.FAKE_HISTORY)
        ),
        *(
            Mutation(Category.FALSE_PREMISE, name, mutate)
            for name, mutate in PREMISE_RULES.items()
        ),
        *(
            Mutation(Category.NEUTRAL, name, mutate)
            for name, mutate in NEUTRAL_RULES.items()
        ),
        *(
            Mutation(
                Category.NEUTRAL,
                item["id"],
                _question_frame(item["text"], is_attack=False),
            )
            for item in _templates(attacks, Category.NEUTRAL)
        ),
    )


def last_user_turn(turns: Sequence[Turn]) -> int | None:
    asked = [place for place, turn in enumerate(turns) if turn.role is Role.USER]
    return asked[-1] if asked else None


def _templates(attacks: dict[str, Any], category: Category) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = attacks[category.value]["templates"]
    return found


def _fills(rng: random.Random) -> dict[str, str]:
    return {
        "attribute": rng.choice(ATTRIBUTES),
        "object": rng.choice(OBJECTS),
        "wrong_value": rng.choice(WRONG_VALUES),
    }


def _kb_insertion(
    template: str, parts: tuple[str, ...], obfuscations: Sequence[Obfuscation]
) -> Mutate:
    def mutate(
        case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
    ) -> tuple[RedTeamCase, Sources]:
        placed = InjectionCase(
            template=template,
            parts=parts,
            document=rng.randrange(len(kb)),
            position=rng.choice(list(Position)),
            obfuscation=rng.choice(list(obfuscations)),
        )
        mutation = kb_injection.mutate(placed, kb, rng)
        mutated = replace(case, planted=mutation.plain, written=mutation.injected)
        return mutated, mutation.sources

    return mutate


def _question_frame(template: str, *, is_attack: bool) -> Mutate:
    """Префикс или суффикс к последней реплике пользователя."""
    frame = tuple(
        piece.strip() for piece in template.split(_QUESTION_SLOT) if piece.strip()
    )

    def mutate(
        case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
    ) -> tuple[RedTeamCase, Sources]:
        question = _with_last_user_turn(
            case.question, lambda asked: template.format(question=asked)
        )
        added = frame if is_attack else ()
        return replace(case, question=question, planted=added, written=added), tuple(kb)

    return mutate


def _fake_history(lines: tuple[str, ...]) -> Mutate:
    def mutate(
        case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
    ) -> tuple[RedTeamCase, Sources]:
        fills = _fills(rng)
        fake = parse_dialogue(
            _LINE_BREAK.join(line.format(**fills) for line in lines)
        ).turns
        turns = list(parse_dialogue(case.question).turns)
        last = last_user_turn(turns)
        if last is None:
            return case, tuple(kb)
        turns[last:last] = fake
        said = tuple(turn.text for turn in fake)
        return (
            replace(case, question=_written(turns), planted=said, written=said),
            tuple(kb),
        )

    return mutate


def _number_shift(
    case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
) -> tuple[RedTeamCase, Sources]:
    """Подменяет в вопросе величину, которую называют и эталон кейса, и база.

    Иначе верное значение для проверки взять неоткуда: вопрос уже может
    содержать ложную предпосылку, а число вроде суммы сделки из вопроса
    («договор на 12 млн») — не утверждение о документах.
    """
    turns = parse_dialogue(case.question).turns
    last = last_user_turn(turns)
    if last is None:
        return case, tuple(kb)
    asked = turns[last].text
    known = find_facts(case.reference)
    in_base = [fact for source in kb for fact in facts_of(source.text)]
    stated = [
        (fact, number)
        for fact in find_facts(asked)
        if isinstance(fact, Quantity)
        and (number := _WHOLE_NUMBER.match(asked, fact.start)) is not None
        and any(same_value(fact, other) for other in known)
        and any(same_value(fact, other) for other in in_base)
    ]
    if not stated:
        return case, tuple(kb)
    fact, number = rng.choice(stated)
    false = _shifted(number.group()) + asked[number.end() : fact.end]
    question = _with_last_user_turn(
        case.question,
        lambda text: f"{text[: fact.start]}{false}{text[fact.end :]}",
    )
    premise = Premise(true=asked[fact.start : fact.end], false=false)
    return replace(case, question=question, premise=premise), tuple(kb)


@functools.cache
def facts_of(text: str) -> tuple[Fact, ...]:
    """Величины текста документа; текст базы разбирается один раз на прогон."""
    return find_facts(text)


def same_value(first: Fact, second: Fact) -> bool:
    """То же число в тех же единицах; оператор и НДС в вопросе и в эталоне
    пишут по-разному («до 7%» и «7%»), поэтому здесь они не сравниваются."""
    return (
        isinstance(first, Quantity)
        and isinstance(second, Quantity)
        and (first.value, first.upper, first.unit)
        == (second.value, second.upper, second.unit)
    )


def _shifted(written: str) -> str:
    """«500 000» → «550 000», «10» → «11»: разделитель разрядов сохраняется."""
    separators = [ch for ch in written if not ch.isdigit()]
    value = int("".join(ch for ch in written if ch.isdigit()))
    digits = str(value + max(1, value // _SHIFT_DIVISOR))
    if not separators:
        return digits
    head = len(digits) % _DIGIT_GROUP or _DIGIT_GROUP
    groups = [digits[:head]] + [
        digits[start : start + _DIGIT_GROUP]
        for start in range(head, len(digits), _DIGIT_GROUP)
    ]
    return separators[0].join(groups)


def _rename_files(
    case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
) -> tuple[RedTeamCase, Sources]:
    # Порядок файлов прежний: его меняет отдельное правило shuffle_order.
    return case, tuple(
        replace(source, name=f"{place:03d}{PurePosixPath(source.name).suffix}")
        for place, source in enumerate(kb, start=1)
    )


def _shuffle_order(
    case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
) -> tuple[RedTeamCase, Sources]:
    return case, tuple(rng.sample(list(kb), len(kb)))


def _whitespace_noise(
    case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
) -> tuple[RedTeamCase, Sources]:
    """Пустые строки между строками файла и пробелы в конце строк."""

    def noisy(text: str) -> str:
        lines: list[str] = []
        for line in text.splitlines():
            lines.append(line + rng.choice(("", " ", "  ")))
            if rng.randrange(_BLANK_LINE_EVERY) == 0:
                lines.append("")
        return _LINE_BREAK.join(lines) + _LINE_BREAK

    return case, tuple(replace(source, text=noisy(source.text)) for source in kb)


def _question_whitespace(
    case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
) -> tuple[RedTeamCase, Sources]:
    question = _with_last_user_turn(
        case.question,
        lambda asked: "".join(
            rng.choice(_SPACE_RUNS) if ch == " " else ch for ch in asked
        ),
    )
    return replace(case, question=question), tuple(kb)


def _question_case(
    case: RedTeamCase, kb: Sequence[SourceFile], rng: random.Random
) -> tuple[RedTeamCase, Sources]:
    return replace(
        case, question=_with_last_user_turn(case.question, str.upper)
    ), tuple(kb)


# Правила разделов без шаблонов: имя — идентификатор правила в eval/attacks.yaml.
PREMISE_RULES: dict[str, Mutate] = {"number_shift": _number_shift}
NEUTRAL_RULES: dict[str, Mutate] = {
    "rename_files": _rename_files,
    "shuffle_order": _shuffle_order,
    "whitespace_noise": _whitespace_noise,
    "question_whitespace": _question_whitespace,
    "question_case": _question_case,
}
# Правила файла, для которых мутации нет: подмена контрагента и номера пункта
# требует каталога сущностей базы, несуществующий документ — номера, которого
# в каталоге нет. Обе вещи зависят от базы, а не от текста вопроса.
PREMISE_RULES_NOT_IMPLEMENTED = frozenset(
    {"wrong_counterparty", "wrong_provision", "nonexistent_document"}
)


def _with_last_user_turn(question: str, rewrite: Callable[[str], str]) -> str:
    turns = list(parse_dialogue(question).turns)
    last = last_user_turn(turns)
    if last is None:
        return question
    turns[last] = replace(turns[last], text=rewrite(turns[last].text))
    return _written(turns)


def _written(turns: Sequence[Turn]) -> str:
    """Диалог в записи кейса; одиночный вопрос остаётся без метки роли."""
    if len(turns) == 1 and turns[0].role is Role.USER:
        return turns[0].text
    return _LINE_BREAK.join(f"{turn.role.value}: {turn.text}" for turn in turns)
