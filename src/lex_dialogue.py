"""Разбор входа кейса на реплики и то, что из диалога берут поиск и промпт.

Вход кейса целиком написан пользователем: и его реплики, и «реплики
ассистента». Разбор только делит текст на реплики и доверия не добавляет ни
одной: реплика ассистента — не факт и не инструкция, в промпт она идёт блоком
данных, в запрос для поиска не идёт вовсе.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from lex_entity_catalog import Counterparty, DocumentEntry, EntityCatalog
from lex_entity_recognition import Entity, Mention, Resolved, combine_turns, recognize
from lex_event_date import EventDate, UnclearEventDate, read_event_date
from lex_normalization import clean_text


class Role(Enum):
    """Участник диалога; значение — метка, с которой реплика идёт в промпт."""

    USER = "Пользователь"
    ASSISTANT = "Ассистент"


# Как участников называют во входе кейса; регистр не важен. Роли «система»
# нет намеренно: во входе кейса её быть не может, и строка «Система: …»
# остаётся текстом той реплики, внутри которой написана.
ROLE_LABELS = {
    Role.USER: ("пользователь", "user", "юзер", "клиент"),
    Role.ASSISTANT: ("ассистент", "assistant", "бот"),
}
# «Пользователь: Какой срок?», «  USER :вопрос», «Бот:» — метка стоит в начале
# строки; «…оплаты. Ассистент: да» посреди строки реплику не начинает.
_LABELLED_LINE = re.compile(
    rf"\s*(?:(?P<user>{'|'.join(ROLE_LABELS[Role.USER])})"
    rf"|(?P<assistant>{'|'.join(ROLE_LABELS[Role.ASSISTANT])}))"
    r"\s*:\s*(?P<text>.*)",
    re.IGNORECASE,
)
_LINE_BREAK = "\n"
# Сколько раз вопрос повторён в запросе для поиска: его слова должны весить
# больше слов прежних реплик, где речь могла идти о другом (демо 6: сначала
# гарантийный срок, затем приёмка). Значение подобрано на шести диалогах
# golden-набора, цифры — в tests/golden/dialogue.yaml.
QUESTION_WEIGHT = 2


@dataclass(frozen=True, slots=True)
class Turn:
    role: Role
    # Без метки роли; строки многострочной реплики разделены переводом строки.
    text: str


@dataclass(frozen=True, slots=True)
class Dialogue:
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class Question:
    """Последняя реплика пользователя: на неё отвечаем."""

    text: str
    # Реплики до вопроса, по порядку.
    history: tuple[Turn, ...]
    # Реплики ассистента после вопроса: вход кончился не репликой пользователя.
    # Вопрос от этого не считается отвеченным — ответ во входе не проверен.
    replies_after: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class NoUserTurn:
    """Отвечать не на что: вход пуст или состоит из одних реплик ассистента."""


@dataclass(frozen=True, slots=True)
class DialogueEntities:
    # Упоминания из реплик пользователя по порядку. Неоднозначное снято, если
    # одного из кандидатов пользователь сам назвал в другой реплике.
    named_by_user: tuple[Mention, ...]
    # Названное только ассистентом и не оспоренное пользователем: не больше
    # одного контрагента и одного документа. Это подсказка, где искать, а не
    # установленный факт.
    suggested_by_assistant: tuple[Entity, ...]


def parse_dialogue(text: str) -> Dialogue:
    """Реплики входа кейса по порядку; реплики без текста пропускаются.

    Строка без метки продолжает текущую реплику, а до первой метки начинает
    реплику пользователя: одиночный вопрос — диалог из одной реплики.
    """
    turns: list[tuple[Role, list[str]]] = []
    for line in clean_text(text).splitlines():
        labelled = _LABELLED_LINE.fullmatch(line)
        if labelled is not None:
            role = Role.USER if labelled["user"] is not None else Role.ASSISTANT
            turns.append((role, [labelled["text"]]))
        elif turns:
            turns[-1][1].append(line)
        elif line.strip():
            turns.append((Role.USER, [line]))
    spoken = ((role, _LINE_BREAK.join(lines).strip()) for role, lines in turns)
    return Dialogue(tuple(Turn(role, said) for role, said in spoken if said))


def last_question(dialogue: Dialogue) -> Question | NoUserTurn:
    asked = [
        position
        for position, turn in enumerate(dialogue.turns)
        if turn.role is Role.USER
    ]
    if not asked:
        return NoUserTurn()
    last = asked[-1]
    return Question(
        text=dialogue.turns[last].text,
        history=dialogue.turns[:last],
        replies_after=dialogue.turns[last + 1 :],
    )


def dialogue_entities(question: Question, catalog: EntityCatalog) -> DialogueEntities:
    """Контрагенты и документы диалога; слово пользователя важнее слова ассистента."""
    named_by_user = combine_turns(
        [recognize(said, catalog) for said in _user_turns(question)]
    )
    by_assistant = _resolved(
        mention
        for turn in (*question.history, *question.replies_after)
        if turn.role is Role.ASSISTANT
        for mention in recognize(turn.text, catalog)
    )
    return DialogueEntities(
        named_by_user=named_by_user,
        suggested_by_assistant=_unchallenged(by_assistant, _resolved(named_by_user)),
    )


def dialogue_event_date(
    question: Question, today: date
) -> EventDate | UnclearEventDate | None:
    """Дата события из реплик пользователя: сначала вопрос, затем прежние реплики
    от поздних к ранним.

    Реплики ассистента не читаются: в них стоят даты договоров и редакций, а
    дату события ассистент знает только со слов пользователя.
    """
    for said in reversed(_user_turns(question)):
        reading = read_event_date(said, today)
        if reading is not None:
            return reading
    return None


def search_query(question: Question, entities: DialogueEntities) -> str:
    """Слова пользователя из всего диалога и названия сущностей, подсказанных
    ассистентом: «а за сколько дней проверить качество?» ищется в договоре,
    о котором шла речь.

    Текст реплик ассистента в запрос не входит: от названной им сущности
    берётся название из каталога, а не его слова. Одиночный вопрос уходит
    в поиск как есть.
    """
    *earlier, asked = _user_turns(question)
    names = [_catalog_name(entity) for entity in entities.suggested_by_assistant]
    weight = QUESTION_WEIGHT if earlier else 1
    return _LINE_BREAK.join([*earlier, *([asked] * weight), *names])


def prompt_history(question: Question) -> tuple[str, ...]:
    """Реплики вокруг вопроса в виде, который принимает `assemble_prompt`.

    Метку роли ставит наш код; `assemble_prompt` заменит её знаками ‹ › вместе
    с метками, написанными внутри текста реплик.
    """
    return tuple(
        f"{turn.role.value}: {turn.text}"
        for turn in (*question.history, *question.replies_after)
    )


def _user_turns(question: Question) -> list[str]:
    """Тексты реплик пользователя по порядку; последний — сам вопрос."""
    earlier = [turn.text for turn in question.history if turn.role is Role.USER]
    return [*earlier, question.text]


def _resolved(mentions: Iterable[Mention]) -> tuple[Entity, ...]:
    return tuple(
        dict.fromkeys(
            mention.resolution.entity
            for mention in mentions
            if isinstance(mention.resolution, Resolved)
        )
    )


def _unchallenged(
    by_assistant: tuple[Entity, ...], by_user: tuple[Entity, ...]
) -> tuple[Entity, ...]:
    """Сущности ассистента, которым пользователь не возразил.

    Сущность берётся, если ассистент назвал её одну такого рода (перечисление
    «„Альфа-Трейд“ или „Альфа-Трейдинг“?» — вопрос, а не утверждение), если
    пользователь сам не назвал контрагента или документ и если она не спорит
    с остальными: договор заключён с названным контрагентом.
    """
    sole: list[Entity] = []
    for kind in (Counterparty, DocumentEntry):
        of_kind = [entity for entity in by_assistant if isinstance(entity, kind)]
        chosen_by_user = any(isinstance(entity, kind) for entity in by_user)
        if len(of_kind) == 1 and not chosen_by_user:
            sole.extend(of_kind)
    return tuple(
        entity
        for entity in sole
        if not any(_are_unrelated(entity, other) for other in (*by_user, *sole))
    )


def _are_unrelated(one: Entity, other: Entity) -> bool:
    """Контрагент и документ, стороной которого он не является."""
    if isinstance(one, DocumentEntry) and isinstance(other, Counterparty):
        return one.id not in other.documents
    if isinstance(one, Counterparty) and isinstance(other, DocumentEntry):
        return other.id not in one.documents
    return False


def _catalog_name(entity: Entity) -> str:
    return entity.name if isinstance(entity, Counterparty) else entity.title
