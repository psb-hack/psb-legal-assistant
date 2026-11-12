"""Генератор вставок в базу по шаблонам eval/attacks.yaml (раздел kb_injection).

Не тест, а общий код состязательного набора: mutate(case, kb, rng) портит один
документ базы и сообщает, какой текст вставлен. Скрытый текст DOCX сюда не
входит: это не мутация текста, он проверяется в тестах чтения DOCX.
"""

from __future__ import annotations

import base64
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import yaml

from lex_documents import Document, DocumentId
from lex_knowledge_base import KnowledgeBase, SourceFile, read_document
from lex_normalization import clean_text
from lex_sanitizer import Screening, admitted_text, default_detectors, screen
from lex_settings import SanitizerThresholds

ATTACKS = Path(__file__).resolve().parents[1] / "eval" / "attacks.yaml"

ATTRIBUTES = ("срок оплаты", "размер неустойки", "срок испытания")
OBJECTS = ("договор поставки № Т-01/2030", "любой договор")
WRONG_VALUES = ("120 календарных дней", "0,5 процента", "два года")

_ZERO_WIDTH_SPACE = "​"
_LATIN_LOOKALIKES = str.maketrans("аеорсх", "aeopcx")
_WORD = re.compile(r"[^\W\d_]+")
_SPLIT_WORDS_FROM_LENGTH = 6


class Position(Enum):
    START = "start"
    BETWEEN_PROVISIONS = "between_provisions"
    AFTER_SIGNATURES = "after_signatures"
    INSIDE_PROVISION = "inside_provision"


class Obfuscation(Enum):
    NONE = "none"
    ZERO_WIDTH = "zero_width"
    HOMOGLYPHS = "homoglyphs"
    BASE64_BLOCK = "base64_block"
    HTML_COMMENT = "html_comment"
    EXTRA_WHITESPACE = "extra_whitespace"


@dataclass(frozen=True, slots=True)
class InjectionCase:
    template: str
    # Абзацы вставки с подстановками {attribute}, {object}, {wrong_value}.
    parts: tuple[str, ...]
    # Номер файла базы, в который идёт вставка.
    document: int
    position: Position
    obfuscation: Obfuscation


@dataclass(frozen=True, slots=True)
class Mutation:
    sources: tuple[SourceFile, ...]
    # Куски вставки в том виде, в каком они записаны в файл.
    injected: tuple[str, ...]
    # Абзацы вставки до маскировки: по ним вставку узнают в ответе.
    plain: tuple[str, ...]


def load_templates() -> dict[str, tuple[str, ...]]:
    section = yaml.safe_load(ATTACKS.read_text(encoding="utf-8"))["kb_injection"]
    return {
        item["id"]: tuple(item.get("parts", [item.get("text")]))
        for item in section["templates"]
    }


def all_cases(documents: int) -> list[InjectionCase]:
    return [
        InjectionCase(template, parts, document, position, obfuscation)
        for template, parts in load_templates().items()
        for document in range(documents)
        for position in Position
        for obfuscation in Obfuscation
    ]


def mutate(
    case: InjectionCase, kb: Sequence[SourceFile], rng: random.Random
) -> Mutation:
    fills = {
        "attribute": rng.choice(ATTRIBUTES),
        "object": rng.choice(OBJECTS),
        "wrong_value": rng.choice(WRONG_VALUES),
    }
    plain = tuple(part.format(**fills) for part in case.parts)
    injected = [
        piece for part in plain for piece in _obfuscated(part, case.obfuscation, rng)
    ]
    target = kb[case.document]
    lines = target.text.splitlines()
    middle = len(lines) // 2
    if case.position is Position.START:
        lines[0:0] = injected
    elif case.position is Position.BETWEEN_PROVISIONS:
        lines[middle:middle] = injected
    elif case.position is Position.AFTER_SIGNATURES:
        lines.extend(injected)
    else:
        lines[middle] = f"{lines[middle]} {injected[0]}"
        lines[middle + 1 : middle + 1] = injected[1:]
    mutated = replace(target, text="\n".join(lines) + "\n")
    sources = (*kb[: case.document], mutated, *kb[case.document + 1 :])
    return Mutation(sources=sources, injected=tuple(injected), plain=plain)


@dataclass(frozen=True, slots=True)
class Outcome:
    case: InjectionCase
    # Хотя бы кусок вставки остался в тексте, который пойдёт в индекс.
    leaked: bool
    # Абзацы, попавшие в карантин из-за вставки, хотя её текста в них нет.
    collateral: tuple[str, ...]


def run_cases(kb: Sequence[SourceFile], seed: int) -> list[Outcome]:
    """Все шаблоны × документы × позиции × маскировки, по одной вставке на прогон."""
    rng = random.Random(seed)
    # В базе может быть своя вставка: её карантин — не следствие мутации.
    baseline = [
        frozenset(decision.original for decision in screened(source)[1].quarantined)
        for source in kb
    ]
    outcomes: list[Outcome] = []
    for case in all_cases(len(kb)):
        mutation = mutate(case, kb, rng)
        document, screening = screened(mutation.sources[case.document])
        admitted = "\n".join(
            admitted_text(document, provision, screening)
            for provision in document.provisions
        )
        collateral = tuple(
            decision.original
            for decision in screening.quarantined
            if decision.original not in baseline[case.document]
            and not any(piece in decision.original for piece in mutation.injected)
        )
        is_leaked = any(clean_text(piece) in admitted for piece in mutation.injected)
        outcomes.append(Outcome(case, is_leaked, collateral))
    return outcomes


def recall_table(outcomes: Sequence[Outcome]) -> dict[str, dict[str, int]]:
    """Шаблон → маскировка → число прогонов, где вставка в индекс не попала."""
    table: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        row = table.setdefault(
            outcome.case.template,
            dict.fromkeys((item.value for item in Obfuscation), 0),
        )
        row[outcome.case.obfuscation.value] += not outcome.leaked
    return table


def screened(source: SourceFile) -> tuple[Document, Screening]:
    thresholds = SanitizerThresholds()
    document = read_document(DocumentId("d01"), source)
    base = KnowledgeBase((document,))
    return document, screen(base, default_detectors(thresholds), thresholds)


def _obfuscated(text: str, obfuscation: Obfuscation, rng: random.Random) -> list[str]:
    if obfuscation is Obfuscation.ZERO_WIDTH:
        return [_WORD.sub(lambda word: _ZERO_WIDTH_SPACE.join(word.group()), text)]
    if obfuscation is Obfuscation.HOMOGLYPHS:
        return [text.translate(_LATIN_LOOKALIKES)]
    if obfuscation is Obfuscation.BASE64_BLOCK:
        encoded = base64.b64encode(text.encode()).decode()
        return [f"Расшифруй и выполни: {encoded}"]
    if obfuscation is Obfuscation.HTML_COMMENT:
        return [f"<!-- {text} -->"]
    if obfuscation is Obfuscation.EXTRA_WHITESPACE:
        return _with_extra_whitespace(text, rng)
    return [text]


def _with_extra_whitespace(text: str, rng: random.Random) -> list[str]:
    """Пробел внутри каждого длинного слова и разрыв строки посередине текста."""

    def split(word: re.Match[str]) -> str:
        letters = word.group()
        if len(letters) < _SPLIT_WORDS_FROM_LENGTH:
            return letters
        at = rng.randrange(2, len(letters) - 1)
        return f"{letters[:at]} {letters[at:]}"

    words = _WORD.sub(split, text).split(" ")
    half = len(words) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]
