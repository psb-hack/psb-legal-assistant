"""Распознавание упоминаний контрагентов и документов в тексте вопроса.

Результат даётся на каждое упоминание, а не на вопрос: «сравни П-17/2024 и
договор с «Альфа-Трейдингом»» называет две разные сущности. Неоднозначность —
отдельное значение с кандидатами; выбор «первого подходящего» здесь запрещён.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import TypeAlias

from rapidfuzz.distance import OSA

from lex_document_header import LEGAL_FORMS
from lex_document_numbers import NumberMention, find_document_numbers
from lex_entity_catalog import Counterparty, DocumentEntry, EntityCatalog, NameToken
from lex_normalization import clean_text, comparison_key

_WORD = re.compile(r"[^\W_]+")
# «Альфа», "Альфа", „Альфа“; из вложенных кавычек находится внутренняя пара.
_QUOTED = r'[«„“"]([^«»„“”"]+)[»“”"]'
_QUOTED_SEGMENT = re.compile(_QUOTED)
# «ООО «Бета-Трейд»»: форма и кавычки называют организацию, даже когда её нет в базе.
_NAMED_ORGANISATION = re.compile(rf"(?<![^\W_])(?:{'|'.join(LEGAL_FORMS)})\s+{_QUOTED}")
# «ИНН 7701234567»: 10 цифр у организации, 12 у предпринимателя. Контрольная
# сумма не проверяется: номер сверяется с каталогом, а не с правилами ФНС.
_TAXPAYER_NUMBER = re.compile(r"(?<!\d)(?:\d{12}|\d{10})(?!\d)")
_WORD_JOINTS = ("", "-")

# Правок на слово по длине его именительного падежа: до 5 букв опечатка
# неотличима от другого слова, от 9 букв допустимы две.
_MIN_FUZZY_LENGTH = 5
_LONG_WORD_LENGTH = 9
# Нечёткий кандидат один, только если остальные дальше лучшего на две правки:
# «Альфа-Трейдин» в одной правке от «Трейдинг» и в двух от «Трейд».
_FUZZY_MARGIN = 2


class Evidence(Enum):
    """Чем подтверждено упоминание; меньшее значение — сильнее свидетельство."""

    # ИНН или номер документа в том виде, в каком он стоит в шапке.
    IDENTIFIER = 1
    FULL_NAME = 2
    # Часть наименования («Альфа») или номер без года («П-17»).
    PARTIAL = 3
    # Наименование с опечаткой: ответ обязан назвать контрагента полностью.
    FUZZY = 4


Entity: TypeAlias = Counterparty | DocumentEntry


@dataclass(frozen=True, slots=True)
class Resolved:
    entity: Entity
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class Ambiguous:
    candidates: tuple[Entity, ...]
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class NotFound:
    """Названа организация или номер документа, которых в базе нет."""

    # Документы с тем же префиксом и числом, но другим годом.
    similar: tuple[DocumentEntry, ...]


Resolution: TypeAlias = Resolved | Ambiguous | NotFound


@dataclass(frozen=True, slots=True)
class Mention:
    # Как написано в тексте: «Северным Складом», «п17/2024».
    text: str
    resolution: Resolution


@dataclass(frozen=True, slots=True)
class _Word:
    start: int
    end: int
    key: str
    capitalized: bool
    # Номер пары кавычек, внутри которой стоит слово.
    quote: int | None


@dataclass(frozen=True, slots=True)
class _Alias:
    tokens: tuple[NameToken, ...]
    counterparty: Counterparty
    full: bool


@dataclass(frozen=True, slots=True)
class _NameMatch:
    start: int
    end: int
    alias: _Alias
    inflected: bool


@dataclass(frozen=True, slots=True)
class _Found:
    start: int
    end: int
    resolution: Resolution


def recognize(text: str, catalog: EntityCatalog) -> tuple[Mention, ...]:
    """Упоминания в порядке появления; пустой кортеж — в тексте никто не назван.

    Текст — одна реплика или вопрос целиком. Диалог распознаётся по репликам,
    результаты объединяет `combine_turns`.
    """
    cleaned = clean_text(text)
    words = _words(cleaned)
    aliases = _aliases(catalog)
    exact = _without_overlaps(
        [*_identifiers(cleaned, catalog), *_exact_names(cleaned, words, aliases)]
    )
    # Опечатка рассматривается только там, где точного совпадения нет.
    fuzzy = _without_overlaps(
        [
            found
            for found in _fuzzy_names(cleaned, words, aliases)
            if not _overlaps_any(found, exact)
        ]
    )
    unknown = [
        found
        for found in _named_organisations(cleaned)
        if not _overlaps_any(found, [*exact, *fuzzy])
    ]
    in_order = sorted([*exact, *fuzzy, *unknown], key=lambda found: found.start)
    return tuple(
        Mention(cleaned[found.start : found.end], found.resolution)
        for found in in_order
    )


def combine_turns(turns: Sequence[tuple[Mention, ...]]) -> tuple[Mention, ...]:
    """Упоминания нескольких реплик подряд.

    Неоднозначное упоминание снимается, когда ровно один из его кандидатов
    назван в этих репликах однозначно: после «с «Альфой»» и «Альфа-Трейдинг»
    остаётся второе. Каким репликам верить, решает вызывающий: он передаёт
    только их.
    """
    named = [
        mention.resolution.entity
        for turn in turns
        for mention in turn
        if isinstance(mention.resolution, Resolved)
    ]

    def is_settled(mention: Mention) -> bool:
        resolution = mention.resolution
        return (
            isinstance(resolution, Ambiguous)
            and sum(candidate in named for candidate in resolution.candidates) == 1
        )

    return tuple(
        mention for turn in turns for mention in turn if not is_settled(mention)
    )


def _words(text: str) -> tuple[_Word, ...]:
    quoted = [match.span(1) for match in _QUOTED_SEGMENT.finditer(text)]

    def quote_of(position: int) -> int | None:
        for index, (start, end) in enumerate(quoted):
            if start <= position < end:
                return index
        return None

    return tuple(
        _Word(
            start=match.start(),
            end=match.end(),
            key=comparison_key(match.group()),
            capitalized=match.group()[0].isupper(),
            quote=quote_of(match.start()),
        )
        for match in _WORD.finditer(text)
    )


def _aliases(catalog: EntityCatalog) -> tuple[_Alias, ...]:
    """Наименование целиком и его части из слов подряд.

    Часть из одних обычных слов алиасом не становится: «склад» не называет
    «Северный Склад».
    """
    aliases: list[_Alias] = []
    for counterparty in catalog.counterparties:
        tokens = counterparty.name_tokens
        for first in range(len(tokens)):
            for end in range(first + 1, len(tokens) + 1):
                part = tokens[first:end]
                full = len(part) == len(tokens)
                if full or not all(token.common for token in part):
                    aliases.append(_Alias(part, counterparty, full))
    return tuple(aliases)


def _windows(
    text: str, words: tuple[_Word, ...], length: int
) -> Iterator[tuple[_Word, ...]]:
    """Слова подряд, которые могут быть одним наименованием.

    Между словами — только пробел или дефис. Окно не разрывает слово через
    дефис и текст в кавычках: «Альфа» в «Альфа-Банк» и в «Альфа Банк» в
    кавычках — часть другого наименования.
    """
    for first in range(len(words) - length + 1):
        window = words[first : first + length]
        if not all(
            _gap(text, left, right) in _WORD_JOINTS for left, right in pairwise(window)
        ):
            continue
        before = words[first - 1] if first > 0 else None
        after = words[first + length] if first + length < len(words) else None
        if before is not None and _is_bound(text, before, window[0]):
            continue
        if after is not None and _is_bound(text, window[-1], after):
            continue
        yield window


def _gap(text: str, left: _Word, right: _Word) -> str:
    return comparison_key(text[left.end : right.start])


def _is_bound(text: str, left: _Word, right: _Word) -> bool:
    """Два слова — части одного: через дефис без пробелов или в одних кавычках."""
    same_quote = left.quote is not None and left.quote == right.quote
    hyphenated = right.start - left.end == 1 and _gap(text, left, right) == "-"
    return same_quote or hyphenated


def _is_emphasized(window: tuple[_Word, ...]) -> bool:
    return any(word.capitalized or word.quote is not None for word in window)


def _is_admitted(alias: _Alias, window: tuple[_Word, ...]) -> bool:
    """Без заглавной буквы и кавычек контрагента называет только полное
    наименование из нескольких слов, среди которых есть необычное.

    «альфа», «логистика», «горизонт» строчными — чаще обычные слова, даже
    когда в текстах базы они так не встречаются.
    """
    if _is_emphasized(window):
        return True
    distinctive = not all(token.common for token in alias.tokens)
    return alias.full and len(alias.tokens) > 1 and distinctive


def _exact_names(
    text: str, words: tuple[_Word, ...], aliases: tuple[_Alias, ...]
) -> list[_Found]:
    matches = [
        _NameMatch(
            start=window[0].start,
            end=window[-1].end,
            alias=alias,
            inflected=any(
                word.key != token.nominative
                for word, token in zip(window, alias.tokens, strict=True)
            ),
        )
        for alias in aliases
        for window in _windows(text, words, len(alias.tokens))
        if _is_admitted(alias, window)
        and all(
            word.key in token.forms
            for word, token in zip(window, alias.tokens, strict=True)
        )
    ]
    spans = dict.fromkeys((match.start, match.end) for match in matches)
    return [
        _Found(*span, _strongest([m for m in matches if (m.start, m.end) == span]))
        for span in spans
    ]


def _strongest(matches: list[_NameMatch]) -> Resolution:
    """Полное наименование сильнее части, именительный падеж — косвенного.

    «Логистика» — это и ООО «Логистика», и родительный падеж ООО «Логистик»;
    названо первое.
    """

    def strength(match: _NameMatch) -> tuple[int, bool]:
        return _name_evidence(match.alias).value, match.inflected

    best = min(strength(match) for match in matches)
    strongest = [match.alias for match in matches if strength(match) == best]
    return _resolution(
        [alias.counterparty for alias in strongest], _name_evidence(strongest[0])
    )


def _name_evidence(alias: _Alias) -> Evidence:
    return Evidence.FULL_NAME if alias.full else Evidence.PARTIAL


def _fuzzy_names(
    text: str, words: tuple[_Word, ...], aliases: tuple[_Alias, ...]
) -> Iterator[_Found]:
    for length in sorted({len(alias.tokens) for alias in aliases}):
        same_length = [alias for alias in aliases if len(alias.tokens) == length]
        for window in _windows(text, words, length):
            resolution = _fuzzy_resolution(window, same_length)
            if resolution is not None:
                yield _Found(window[0].start, window[-1].end, resolution)


def _fuzzy_resolution(
    window: tuple[_Word, ...], aliases: list[_Alias]
) -> Resolution | None:
    distances = [
        (alias, _edit_distances(alias, window))
        for alias in aliases
        if _is_admitted(alias, window)
    ]
    within_limits = [
        sum(edits)
        for alias, edits in distances
        if all(
            edit <= _edit_limit(token)
            for edit, token in zip(edits, alias.tokens, strict=True)
        )
    ]
    if not within_limits:
        return None
    nearest = min(within_limits)
    rivals = [
        alias.counterparty
        for alias, edits in distances
        if sum(edits) < nearest + _FUZZY_MARGIN
    ]
    return _resolution(rivals, Evidence.FUZZY)


def _edit_distances(alias: _Alias, window: tuple[_Word, ...]) -> tuple[int, ...]:
    # OSA считает перестановку соседних букв одной правкой: «трейдниг».
    return tuple(
        min(OSA.distance(word.key, form) for form in token.forms)
        for word, token in zip(window, alias.tokens, strict=True)
    )


def _edit_limit(token: NameToken) -> int:
    length = len(token.nominative)
    if length < _MIN_FUZZY_LENGTH:
        return 0
    return 1 if length < _LONG_WORD_LENGTH else 2


def _identifiers(text: str, catalog: EntityCatalog) -> Iterator[_Found]:
    for mention in find_document_numbers(text):
        resolution = _number_resolution(mention, catalog.documents)
        if resolution is not None:
            yield _Found(mention.start, mention.end, resolution)
    for match in _TAXPAYER_NUMBER.finditer(text):
        holders = [
            counterparty
            for counterparty in catalog.counterparties
            if counterparty.taxpayer_number == match.group()
        ]
        if holders:
            yield _Found(*match.span(), _resolution(holders, Evidence.IDENTIFIER))


def _number_resolution(
    mention: NumberMention, documents: tuple[DocumentEntry, ...]
) -> Resolution | None:
    series = [
        (entry, entry.number.year)
        for entry in documents
        if entry.number is not None and entry.number.same_series(mention.number)
    ]
    same_year = [entry for entry, year in series if year == mention.number.year]
    similar = [entry for entry, _ in series]
    if same_year:
        return _resolution(same_year, Evidence.IDENTIFIER)
    if mention.number.year is None:
        return _resolution(similar, Evidence.PARTIAL) if similar else None
    # Год назван и не совпал: это другой документ, а не тот же с опечаткой.
    if similar or mention.written_in_full:
        return NotFound(similar=tuple(similar))
    return None


def _named_organisations(text: str) -> Iterator[_Found]:
    for match in _NAMED_ORGANISATION.finditer(text):
        yield _Found(*match.span(), NotFound(similar=()))


def _resolution(entities: Sequence[Entity], evidence: Evidence) -> Resolution:
    distinct = tuple(dict.fromkeys(entities))
    if len(distinct) == 1:
        return Resolved(distinct[0], evidence)
    return Ambiguous(distinct, evidence)


def _without_overlaps(found: list[_Found]) -> list[_Found]:
    """Самое длинное совпадение гасит вложенные: «Альфа-Трейдинг» гасит «Альфу»."""
    kept: list[_Found] = []
    for item in sorted(found, key=lambda item: (item.start - item.end, item.start)):
        if not _overlaps_any(item, kept):
            kept.append(item)
    return kept


def _overlaps_any(item: _Found, others: Sequence[_Found]) -> bool:
    return any(item.start < other.end and other.start < item.end for other in others)
