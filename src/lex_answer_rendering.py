"""Сборка текста ответа из проверенного структурного ответа модели.

Модель даёт утверждения и идентификаторы положений. Первую фразу, ссылки,
фразы об отсутствии документов и отказы пишет код. Юридических утверждений
в шаблонах нет: они обрамляют то, что пришло из положений.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby

from lex_answer_contract import AnswerVerdict, GroundedAnswer, GroundedClaim
from lex_documents import Document, Provision, ProvisionId, cite
from lex_errors import VerificationError
from lex_knowledge_base import KnowledgeBase
from lex_scenarios import Scenario

# Контракт ответа, docs/03: «до 150 слов».
MAX_ANSWER_WORDS = 150

VERDICT_LEADS: Mapping[AnswerVerdict, str] = {
    AnswerVerdict.YES: "Да.",
    AnswerVerdict.NO: "Нет.",
}
SOURCE_LINE = "Источник: {reference}."
NO_DOCUMENTS_ABOUT = "В базе знаний нет документов {topics}."
REFUSAL_ABOUT = (
    "В базе знаний нет документов {topics}, поэтому ответить по документам "
    "я не могу. По памяти отвечать не буду."
)
REFUSAL_NO_DATA = (
    "В базе знаний нет документов, по которым можно ответить на этот вопрос. "
    "По памяти отвечать не буду."
)
CLARIFICATION_REQUEST = "Уточните, пожалуйста: {details}."
ILLEGAL_LEAD = "Помочь с этим не могу."
ROLE_REPLY = (
    "Я остаюсь юридическим ассистентом и отвечаю только по документам базы знаний."
)
OFF_TOPIC_REPLY = (
    "Я отвечаю на вопросы по нормативным документам и договорам из базы знаний. "
    "Переформулируйте, пожалуйста, вопрос так, чтобы он относился к этим документам."
)

_ABOUT = "о"
_ABOUT_BEFORE_VOWEL = "об"
# «об аренде», «об условиях», но «о единой ставке», «о юристе»: е, ё, ю, я
# в начале слова читаются с согласного «й».
_VOWELS_TAKING_OB = frozenset("аиоуэ")
# «о пособиях», «Об отпуске», «обо всех доплатах»: предлог уже поставила модель
_LEADING_ABOUT = re.compile(r"^(?:обо|об|о)\s+", re.IGNORECASE)
_SENTENCE_END = ".!?…"
_FULL_STOP = "."


@dataclass(frozen=True, slots=True)
class _Passage:
    """Утверждения подряд с одной опорой: ссылка ставится один раз, после них."""

    sentences: tuple[str, ...]
    sources: tuple[tuple[Document, Provision], ...]


def render(answer: GroundedAnswer, base: KnowledgeBase) -> str:
    """Текст ответа по контракту docs/03: прямой ответ, утверждения, границы.

    Ссылку получает только положение из утверждений answer: текст ссылок
    модель не пишет.
    """
    if answer.scenario is Scenario.OFF_TOPIC:
        return OFF_TOPIC_REPLY
    claims = answer.claims
    text = _composed(answer, claims, base)
    # Сокращаем с конца: первые утверждения отвечают на вопрос, последние дополняют.
    while len(text.split()) > MAX_ANSWER_WORDS and len(claims) > 1:
        claims = claims[:-1]
        text = _composed(answer, claims, base)
    return text


def _composed(
    answer: GroundedAnswer, claims: Sequence[GroundedClaim], base: KnowledgeBase
) -> str:
    parts = [
        *_opening(answer),
        *_supported(_passages(claims, base)),
        *_limits(answer),
    ]
    return " ".join(parts) if parts else REFUSAL_NO_DATA


def _opening(answer: GroundedAnswer) -> list[str]:
    if answer.scenario is Scenario.ILLEGAL:
        return [ILLEGAL_LEAD]
    if not answer.claims:
        return [ROLE_REPLY] if answer.scenario is Scenario.ROLE_ATTACK else []
    lead = VERDICT_LEADS.get(answer.verdict)
    return [] if lead is None else [lead]


def _limits(answer: GroundedAnswer) -> list[str]:
    if not answer.missing:
        return []
    if answer.scenario is Scenario.AMBIGUOUS:
        details = "; ".join(_bare(item) for item in answer.missing)
        return [CLARIFICATION_REQUEST.format(details=details)]
    topics = _enumerated([_about(item) for item in answer.missing])
    if answer.claims or _opening(answer):
        return [NO_DOCUMENTS_ABOUT.format(topics=topics)]
    return [REFUSAL_ABOUT.format(topics=topics)]


def _passages(
    claims: Sequence[GroundedClaim], base: KnowledgeBase
) -> tuple[_Passage, ...]:
    passages: list[_Passage] = []
    for _, group in groupby(claims, key=lambda claim: frozenset(claim.provisions)):
        first, *rest = group
        passages.append(
            _Passage(
                sentences=tuple(_sentence(claim.text) for claim in (first, *rest)),
                sources=tuple(_found(cited, base) for cited in first.provisions),
            )
        )
    return tuple(passages)


def _supported(passages: Sequence[_Passage]) -> list[str]:
    if len(passages) == 1:
        # Опора одна на весь ответ: ссылка после каждой фразы повторяла бы её.
        (only,) = passages
        reference = _reference(only.sources, named=None)
        return [*only.sentences, SOURCE_LINE.format(reference=reference)]
    sentences: list[str] = []
    named: Document | None = None
    for passage in passages:
        *before, last = passage.sentences
        reference = _reference(passage.sources, named)
        sentences.extend(before)
        sentences.append(f"{last.rstrip(_SENTENCE_END)} ({reference}).")
        named = passage.sources[-1][0]
    return sentences


def _reference(
    sources: Sequence[tuple[Document, Provision]], named: Document | None
) -> str:
    """Ссылка на положения: документ назван один раз, дальше только адреса.

    named — документ, названный предыдущей ссылкой: адрес без названия читатель
    отнесёт к нему.
    """
    parts: list[str] = []
    previous = named
    for _, group in groupby(sources, key=lambda source: source[0].id):
        (document, first), *rest = group
        same_document = previous is not None and previous.id == document.id
        head = first.label if same_document else cite(document, first)
        labels = dict.fromkeys(provision.label for _, provision in rest)
        labels.pop(first.label, None)
        parts.append(", ".join([head, *labels]))
        previous = document
    return "; ".join(parts)


def _found(
    provision_id: ProvisionId, base: KnowledgeBase
) -> tuple[Document, Provision]:
    found = base.find(provision_id)
    if found is None:
        raise VerificationError(f"положения {provision_id} нет в базе")
    return found


def _sentence(text: str) -> str:
    stripped = text.strip()
    return (
        stripped if stripped.endswith(tuple(_SENTENCE_END)) else stripped + _FULL_STOP
    )


def _bare(item: str) -> str:
    return item.strip().rstrip(_SENTENCE_END).strip()


def _about(topic: str) -> str:
    """«о пособиях», «об увеличении платы»: тема в предложном падеже с предлогом."""
    bare = _bare(topic)
    if _LEADING_ABOUT.match(bare):
        return bare[0].lower() + bare[1:]
    first_letter = next((ch for ch in bare if ch.isalpha()), "")
    takes_ob = first_letter.lower() in _VOWELS_TAKING_OB
    return f"{_ABOUT_BEFORE_VOWEL if takes_ob else _ABOUT} {bare}"


def _enumerated(items: Sequence[str]) -> str:
    *head, last = items
    return f"{', '.join(head)} и {last}" if head else last
