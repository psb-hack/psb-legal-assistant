"""Представления текста базы: чистый текст, ключ сравнения и ключ детекции.

Чистый текст идёт в индекс и в модель, ключ сравнения — в поиск и сверку строк,
ключ детекции — только детекторам вставок: в нём текст намеренно искажён.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

_KEPT_CONTROLS = frozenset("\n\t\r")
# Формат, управляющие, частное использование, не назначенные.
_REMOVED_CATEGORIES = frozenset({"Cf", "Cc", "Co", "Cn"})
# Невидимые символы с категорией буквы, знака или символа: по категории не ловятся.
_INVISIBLE_RANGES = (
    (0xFE00, 0xFE0F),  # селекторы вариантов
    (0xE0100, 0xE01EF),  # селекторы вариантов, дополнение
    (0x115F, 0x1160),  # заполнители хангыля
    (0x3164, 0x3164),
    (0xFFA0, 0xFFA0),
    (0x2800, 0x2800),  # пустой шаблон Брайля
)
# Tag-блок повторяет ASCII невидимыми символами: U+E0041 читается моделью как «A».
_TAG_BLOCK_START = 0xE0000
_TAG_BLOCK_END = 0xE007F
_TAG_PRINTABLE = range(0xE0020, 0xE007F)
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")
# Метка порядка байтов и мягкий перенос встречаются в обычных файлах: не находка.
_UNREMARKABLE_FORMAT = frozenset("﻿­")

_QUOTES = dict.fromkeys(map(ord, "«»„“”‟❝❞＂"), '"') | dict.fromkeys(
    map(ord, "‘’‚‛`´"), "'"
)
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_YO = {ord("ё"): "е"}
_PUNCTUATION_VARIANTS = _QUOTES | _DASHES | _YO
_WHITESPACE_RUN = re.compile(r"\s+")

# «Cрок» с латинской «C» — одно слово; дефис и цифры слово разрывают.
_WORD = re.compile(r"[^\W\d_]+")
# Пары букв, неотличимых на глаз после приведения регистра: кириллица — латиница.
_LOOKALIKE_PAIRS = (
    ("а", "a"),
    ("в", "b"),
    ("е", "e"),
    ("к", "k"),
    ("м", "m"),
    ("н", "h"),
    ("о", "o"),
    ("р", "p"),
    ("с", "c"),
    ("т", "t"),
    ("у", "y"),
    ("х", "x"),
)
_TO_LATIN = dict(_LOOKALIKE_PAIRS)
_TO_CYRILLIC = {latin: cyrillic for cyrillic, latin in _LOOKALIKE_PAIRS}
_SAMPLE_WORDS = 5

# «0KHQutGA0L7QuiDQvtC/0LvQsNGC0Ysg0LjQt9C80LXQvdGR0L0=» — не короче 24 знаков:
# короче бывают номера и обозначения.
_BASE64_BLOCK = re.compile(
    r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])"
)
# Доля букв и пробелов, при которой раскодированное считается текстом, а не шумом.
_READABLE_SHARE = 0.8


class TextFindingKind(Enum):
    INVISIBLE_CHARACTERS = "invisible_characters"
    TAG_CHARACTERS = "tag_characters"
    BIDI_OVERRIDE = "bidi_override"
    MIXED_SCRIPT_WORDS = "mixed_script_words"
    ENCODED_BLOCK = "encoded_block"


@dataclass(frozen=True, slots=True)
class TextFinding:
    """То, что нормализация убрала или раскрыла: данные для детекторов и журнала."""

    kind: TextFindingKind
    count: int
    # Раскодированный текст или слова со смешением алфавитов; иначе пустая строка.
    sample: str


@dataclass(frozen=True, slots=True)
class NormalizedText:
    original: str
    clean: str
    detection_key: str
    findings: tuple[TextFinding, ...]


class _Script(Enum):
    CYRILLIC = "CYRILLIC"
    LATIN = "LATIN"
    OTHER = "OTHER"


def normalize(text: str) -> NormalizedText:
    clean = clean_text(text)
    tag_text = _decoded_tags(text)
    payloads = _decoded_base64(clean)
    folded, mixed_words = _fold_lookalikes(_detection_base(clean))
    hidden_parts = [_detection_base(part) for part in (tag_text, *payloads) if part]
    return NormalizedText(
        original=text,
        clean=clean,
        detection_key=" ".join([folded, *hidden_parts]),
        findings=tuple(_findings(text, tag_text, payloads, mixed_words)),
    )


def clean_text(text: str) -> str:
    """NFC без невидимых и управляющих символов; переводы строк на месте.

    NFKC сюда не подходит: он превращает «№» в «No», а «м²» в «м2».
    """
    composed = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in composed if not _is_removed(ch))


def comparison_key(text: str) -> str:
    """Один вид для поиска и сверки строк: регистр, ё/е, кавычки, тире, пробелы."""
    lowered = clean_text(text).casefold().translate(_PUNCTUATION_VARIANTS)
    return _WHITESPACE_RUN.sub(" ", lowered).strip()


def _detection_base(text: str) -> str:
    return comparison_key(unicodedata.normalize("NFKC", text))


def _is_removed(ch: str) -> bool:
    if ch in _KEPT_CONTROLS:
        return False
    return unicodedata.category(ch) in _REMOVED_CATEGORIES or _in_invisible_range(ch)


def _in_invisible_range(ch: str) -> bool:
    return any(first <= ord(ch) <= last for first, last in _INVISIBLE_RANGES)


def _is_tag(ch: str) -> bool:
    return _TAG_BLOCK_START <= ord(ch) <= _TAG_BLOCK_END


def _is_remarkable_invisible(ch: str) -> bool:
    if _is_tag(ch) or ch in _BIDI_CONTROLS or ch in _UNREMARKABLE_FORMAT:
        return False
    return unicodedata.category(ch) == "Cf" or _in_invisible_range(ch)


def _decoded_tags(text: str) -> str:
    return "".join(
        chr(ord(ch) - _TAG_BLOCK_START) for ch in text if ord(ch) in _TAG_PRINTABLE
    )


def _decoded_base64(text: str) -> tuple[str, ...]:
    return tuple(
        decoded
        for block in _BASE64_BLOCK.findall(text)
        if (decoded := _readable_base64(block)) is not None
    )


def _readable_base64(block: str) -> str | None:
    try:
        decoded = base64.b64decode(block, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    readable = sum(ch.isalpha() or ch.isspace() for ch in decoded)
    if not decoded or readable / len(decoded) < _READABLE_SHARE:
        return None
    return decoded


def _fold_lookalikes(text: str) -> tuple[str, tuple[str, ...]]:
    """Слово со смешением алфавитов приводится к алфавиту большинства его букв.

    Целиком по таблице confusables текст сворачивать нельзя: она отображает
    кириллицу в латиницу, и русский текст перестаёт читаться.
    """
    mixed_words: list[str] = []

    def fold(match: re.Match[str]) -> str:
        word = match.group()
        scripts = [_script(ch) for ch in word]
        cyrillic = scripts.count(_Script.CYRILLIC)
        latin = scripts.count(_Script.LATIN)
        if not cyrillic or not latin:
            return word
        mixed_words.append(word)
        table = _TO_CYRILLIC if cyrillic >= latin else _TO_LATIN
        return "".join(table.get(ch, ch) for ch in word)

    return _WORD.sub(fold, text), tuple(mixed_words)


def _script(ch: str) -> _Script:
    name = unicodedata.name(ch, "")
    for script in (_Script.CYRILLIC, _Script.LATIN):
        if name.startswith(script.value):
            return script
    return _Script.OTHER


def _findings(
    text: str, tag_text: str, payloads: tuple[str, ...], mixed_words: tuple[str, ...]
) -> list[TextFinding]:
    invisible = sum(_is_remarkable_invisible(ch) for ch in text)
    tags = sum(_is_tag(ch) for ch in text)
    bidi = sum(ch in _BIDI_CONTROLS for ch in text)
    counted = (
        (TextFindingKind.INVISIBLE_CHARACTERS, invisible, ""),
        (TextFindingKind.TAG_CHARACTERS, tags, tag_text),
        (TextFindingKind.BIDI_OVERRIDE, bidi, ""),
        (
            TextFindingKind.MIXED_SCRIPT_WORDS,
            len(mixed_words),
            ", ".join(mixed_words[:_SAMPLE_WORDS]),
        ),
        (TextFindingKind.ENCODED_BLOCK, len(payloads), " ".join(payloads)),
    )
    return [
        TextFinding(kind, count, sample) for kind, count, sample in counted if count
    ]
