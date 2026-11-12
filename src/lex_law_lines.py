"""Строки статьи закона из выгрузки справочной системы: что норма, а что нет."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from lex_paragraphs import split_paragraphs

# «1», «13.11», «1.8-3»
UNIT_NUMBER_PATTERN = r"\d+(?:\.\d+)*(?:-\d+)?"

# «Источник: Статья 70. ТК РФ …», «URL: https://…», «Дата выгрузки: 10.10.2026»
_SOURCE_FIELD = re.compile(r"^(?:Источник|URL|Дата\s+выгрузки):")
# «Статья 13.11. Нарушение законодательства…»; пометка «Статья 59 изменена с …»
# сюда не подходит: в ней после номера нет точки.
_ARTICLE_TITLE = re.compile(
    rf"^Статья\s+(?P<number>{UNIT_NUMBER_PATTERN})\.(?:\s+(?P<title>\S.*))?$"
)
# «с 1 сентября 2026 г.» — так год пишет справочная система; в нормах — «года».
_SHORT_YEAR = re.compile(r"\d{4}\s+г\.")
# Норма заканчивается знаком препинания; диспозиция в КоАП — тире перед санкцией.
_NORM_ENDINGS = (".", ";", ":", ",", "-", "—")
# «1.8-3. Налоговая ставка…» и «2.1) тяжелое материальное положение…»
_UNIT_MARKER = re.compile(
    rf"^(?P<number>{UNIT_NUMBER_PATTERN})(?P<closing>[.)])\s+(?P<text>\S.*)$"
)
_REPEAL_VERB = r"(?:утратил[аи]?\s+силу|исключен[аы]?)"
# «4. Утратил силу с 1 января 2015 г.», «2) утратил силу с 1 января 2011 г.;»
_REPEALED_UNIT = re.compile(
    rf"^{UNIT_NUMBER_PATTERN}[.)]\s+{_REPEAL_VERB}", re.IGNORECASE
)
# «Абзац второй исключен.», «абзацы третий - шестой утратили силу с 1 января 2019 г.»
_REPEALED_PARAGRAPHS = re.compile(
    r"^абзацы?(?:\s+(?P<first>\S+)(?:\s+[-–—]\s+(?P<last>\S+))?)?"
    rf"\s+{_REPEAL_VERB}",
    re.IGNORECASE,
)
# «Часть вторая утратила силу.»
_REPEALED_PART = re.compile(rf"^часть(?:\s+\S+)?\s+{_REPEAL_VERB}", re.IGNORECASE)
_ORDINAL_STEMS = (
    "перв",
    "втор",
    "трет",
    "четв[её]рт",
    "пят",
    "шест",
    "седьм",
    "восьм",
    "девят",
    "десят",
)
# «третий», «шестой», «вторая» → основа из _ORDINAL_STEMS и окончание
_ORDINAL_ENDING = r"(?:ый|ой|ий|ая|ья)"


class LawLineKind(Enum):
    SOURCE = "source"
    ARTICLE_TITLE = "article_title"
    NORM = "norm"
    COMMENT = "comment"
    # Строка на месте единицы, утратившей силу.
    REPEALED = "repealed"


class MarkerClosing(Enum):
    DOT = "."
    BRACKET = ")"


@dataclass(frozen=True, slots=True)
class LawLine:
    kind: LawLineKind
    text: str


@dataclass(frozen=True, slots=True)
class ArticleTitle:
    number: str
    # Пустая строка, если после номера статьи сразу идёт норма.
    title: str


@dataclass(frozen=True, slots=True)
class UnitMarker:
    number: str
    closing: MarkerClosing
    text: str


def is_law_article(text: str) -> bool:
    """Выгрузка статьи закона: шапка источника и строка «Статья N.» после неё."""
    kinds = [line.kind for line in classify_lines(split_paragraphs(text))]
    return (
        bool(kinds)
        and kinds[0] is LawLineKind.SOURCE
        and LawLineKind.ARTICLE_TITLE in kinds
    )


def classify_lines(paragraphs: Sequence[str]) -> list[LawLine]:
    """Каждому абзацу — вид строки. Порядок и текст абзацев сохраняются.

    Строка «Статья 1. Установить …» делится на заголовок «Статья 1.» и норму.
    """
    lines: list[LawLine] = []
    title_found = False
    for paragraph in paragraphs:
        title = None if title_found else read_article_title(paragraph)
        if title is not None:
            title_found = True
            lines.extend(_title_lines(paragraph, title))
        elif not title_found and _SOURCE_FIELD.match(paragraph):
            lines.append(LawLine(LawLineKind.SOURCE, paragraph))
        else:
            lines.append(LawLine(_kind_of(paragraph), paragraph))
    return lines


def read_article_title(line: str) -> ArticleTitle | None:
    match = _ARTICLE_TITLE.match(line)
    if match is None:
        return None
    rest = match["title"] or ""
    # Название статьи знаком не заканчивается; иначе после номера идёт норма.
    return ArticleTitle(match["number"], "" if rest.endswith(_NORM_ENDINGS) else rest)


def split_marker(line: str) -> UnitMarker | None:
    match = _UNIT_MARKER.match(line)
    if match is None:
        return None
    return UnitMarker(match["number"], MarkerClosing(match["closing"]), match["text"])


def repealed_paragraph_count(line: str) -> int | None:
    """Сколько абзацев занимает заглушка; None — строка не про абзацы."""
    match = _REPEALED_PARAGRAPHS.match(line)
    if match is None:
        return None
    first = _ordinal(match["first"])
    last = _ordinal(match["last"])
    if first is None or last is None or last < first:
        return 1
    return last - first + 1


def is_repealed_part(line: str) -> bool:
    return _REPEALED_PART.match(line) is not None


def _title_lines(paragraph: str, title: ArticleTitle) -> list[LawLine]:
    match = _ARTICLE_TITLE.match(paragraph)
    if title.title or match is None or match["title"] is None:
        return [LawLine(LawLineKind.ARTICLE_TITLE, paragraph)]
    marker = paragraph[: match.start("title")].rstrip()
    return [
        LawLine(LawLineKind.ARTICLE_TITLE, marker),
        LawLine(_kind_of(match["title"]), match["title"]),
    ]


def _kind_of(line: str) -> LawLineKind:
    is_repealed = (
        _REPEALED_UNIT.match(line)
        or _REPEALED_PARAGRAPHS.match(line)
        or _REPEALED_PART.match(line)
    )
    if is_repealed:
        return LawLineKind.REPEALED
    return LawLineKind.COMMENT if _is_comment(line) else LawLineKind.NORM


def _is_comment(line: str) -> bool:
    return _SHORT_YEAR.search(line) is not None or not line.endswith(_NORM_ENDINGS)


def _ordinal(word: str | None) -> int | None:
    if word is None:
        return None
    if word.isdecimal():
        return int(word)
    for position, stem in enumerate(_ORDINAL_STEMS, start=1):
        if re.fullmatch(stem + _ORDINAL_ENDING, word, re.IGNORECASE):
            return position
    return None
