"""Номера документов: разбор на префикс, число и год, поиск номеров в тексте.

Номер сравнивается по частям и только точно. Нечёткое сравнение здесь вредно:
«П-17/2024» и «П-17/2023» отличаются одним знаком и называют разные документы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lex_normalization import CYRILLIC_LOOKALIKES, DASH_VARIANTS, comparison_key

# Тот, кто набирает номер латиницей, пишет либо похожую букву («A» вместо «А»),
# либо букву с тем же звуком («P» вместо «П»). Латинская «P» по начертанию —
# кириллическая «Р», поэтому проверяются оба прочтения.
_SOUND_ALIKES = str.maketrans("abvgdeziklmnoprstufhc", "абвгдезиклмнопрстуфхц")
_LOOK_ALIKES = str.maketrans(CYRILLIC_LOOKALIKES)

_LETTER = r"[^\W\d_]"
_DASH = f"[-{DASH_VARIANTS}]"
# Границы заданы явно: «\b» не срабатывает между «№» и буквой и рядом с дефисом.
_NOT_AFTER_WORD = r"(?<![^\W_])"
_NUMBER_START = (
    rf"{_NOT_AFTER_WORD}(?P<prefix>{_LETTER}{{1,3}})(?P<joint>\s|{_DASH})?"
    r"(?P<number>\d{1,4})"
)
# «П-17/2024», «п17/2024», «P 17-2024», «п-17/24». Продолжение «…/2024-1» —
# уже другой номер, поэтому после года не должно быть разделителя с цифрой.
_NUMBER_WITH_YEAR = re.compile(
    rf"{_NUMBER_START}(?P<separator>\s?(?:[/\\]|{_DASH})\s?)"
    rf"(?P<year>(?:19|20)\d{{2}}|\d{{2}})(?![^\W_])(?!(?:[/\\]|{_DASH})\d)"
)
# «П-17», «ЮО-7». Без года номером считается только запись через дефис: иначе
# «п. 17» и «склад 3» стали бы номерами документов.
_NUMBER_WITHOUT_YEAR = re.compile(
    rf"{_NOT_AFTER_WORD}(?P<prefix>{_LETTER}{{1,3}}){_DASH}(?P<number>\d{{1,4}})"
    rf"(?![^\W_]|[/\\]|{_DASH})"
)
_FULL_YEAR_LENGTH = 4
# Год из двух цифр относится к текущему веку: «п-17/24» — это 2024.
_CENTURY = 2000


@dataclass(frozen=True, slots=True)
class DocumentNumber:
    # Буквы перед числом в ключе сравнения: «п», «пт», «юо».
    prefix: str
    # Число, а не строка: «П-017» и «П-17» — один номер.
    number: int
    year: int | None

    def same_series(self, other: DocumentNumber) -> bool:
        """Тот же префикс в любом прочтении и то же число; год не сравнивается."""
        return self.number == other.number and not _prefix_readings(
            self.prefix
        ).isdisjoint(_prefix_readings(other.prefix))


@dataclass(frozen=True, slots=True)
class NumberMention:
    start: int
    end: int
    number: DocumentNumber
    # «П-17/2023», «п17-2023»: буквы вплотную к числу и год из четырёх цифр.
    # Такая запись называет документ, даже если его нет в базе; «на 12-20» — нет.
    written_in_full: bool


def parse_document_number(text: str) -> DocumentNumber | None:
    """Номер из шапки документа; None — запись не вида «буквы, число, год»."""
    stripped = text.strip()
    match = _NUMBER_WITH_YEAR.fullmatch(stripped) or _NUMBER_WITHOUT_YEAR.fullmatch(
        stripped
    )
    return None if match is None else _number(match)


def find_document_numbers(text: str) -> tuple[NumberMention, ...]:
    """Записи, похожие на номер документа, в порядке их появления в тексте."""
    with_year = [
        NumberMention(*match.span(), _number(match), _is_written_in_full(match))
        for match in _NUMBER_WITH_YEAR.finditer(text)
        if not _is_loose(match)
    ]
    without_year = [
        NumberMention(*match.span(), _number(match), written_in_full=False)
        for match in _NUMBER_WITHOUT_YEAR.finditer(text)
        if not any(found.start <= match.start() < found.end for found in with_year)
    ]
    return tuple(sorted(with_year + without_year, key=lambda found: found.start))


def _number(match: re.Match[str]) -> DocumentNumber:
    year = match.groupdict().get("year")
    return DocumentNumber(
        prefix=comparison_key(match["prefix"]),
        number=int(match["number"]),
        year=None if year is None else _full_year(year),
    )


def _full_year(year: str) -> int:
    return int(year) if len(year) == _FULL_YEAR_LENGTH else _CENTURY + int(year)


def _is_loose(match: re.Match[str]) -> bool:
    """«а 8-24», «с 5/10»: предлог, диапазон или дробь, а не номер."""
    joint = match["joint"]
    return (
        joint is not None and joint.isspace() and len(match["year"]) < _FULL_YEAR_LENGTH
    )


def _is_written_in_full(match: re.Match[str]) -> bool:
    joint = match["joint"]
    separator = match["separator"]
    return (
        (joint is None or not joint.isspace())
        and separator == separator.strip()
        and len(match["year"]) == _FULL_YEAR_LENGTH
    )


def _prefix_readings(prefix: str) -> frozenset[str]:
    return frozenset(
        {prefix, prefix.translate(_LOOK_ALIKES), prefix.translate(_SOUND_ALIKES)}
    )
