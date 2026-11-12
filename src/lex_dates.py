"""Даты в деловых текстах: «01.03.2025», «1 марта 2025 г.», «в августе 2024 года»."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
_DAY = r"\d{1,2}"
_MONTH_NUMBER = r"\d{2}"
_MONTH_NAME = rf"(?:{'|'.join(_MONTHS)})"
_YEAR = r"\d{4}"
# В свободном тексте «2000 г.» и «2000 рублей» различает только слово после числа,
# поэтому годом считается число из двух ближайших столетий.
_CENTURY_YEAR = r"(?:19|20)\d{2}"
# Именительный, родительный и предложный падежи: «август», «августа», «в августе».
_MONTH_WORDS = (
    "январ[ьяе]",
    "феврал[ьяе]",
    "март[ае]?",
    "апрел[ьяе]",
    "ма[йяе]",
    "июн[ьяе]",
    "июл[ьяе]",
    "август[ае]?",
    "сентябр[ьяе]",
    "октябр[ьяе]",
    "ноябр[ьяе]",
    "декабр[ьяе]",
)
# Без групп, как DATE_PATTERN. Рассчитан на re.IGNORECASE.
MONTH_WORD_PATTERN = rf"(?<![а-яё])(?:{'|'.join(_MONTH_WORDS)})(?![а-яё])"

# «01.03.2025»
_NUMERIC_DATE = re.compile(rf"({_DAY})\.({_MONTH_NUMBER})\.({_YEAR})")
# «1 марта 2025»
_VERBAL_DATE = re.compile(rf"({_DAY})\s+({_MONTH_NAME})\s+({_YEAR})", re.IGNORECASE)

# Без групп: шаблон вставляют в выражения других модулей, иногда дважды в одно.
DATE_PATTERN = (
    rf"(?<!\d)(?:{_DAY}\.{_MONTH_NUMBER}\.{_YEAR}"
    rf"|{_DAY}\s+{_MONTH_NAME}\s+{_YEAR})(?!\d)"
)

# «01.08.2024» | «1 августа 2024», «в августе 2024», «август 2024» | «в 2024 году»,
# «2024 г.». Год без слова «год» датой не считается: «№ П-17/2024», «2024 рубля».
_MENTION = re.compile(
    rf"(?<![\d./-])(?:(?P<numeric>{_DAY}\.{_MONTH_NUMBER}\.{_CENTURY_YEAR})"
    rf"|(?:(?P<day>{_DAY})\s+)?(?P<month>{MONTH_WORD_PATTERN})\s+(?P<year>{_CENTURY_YEAR})"
    rf"|(?P<bare_year>{_CENTURY_YEAR})\s*(?:г\.|год(?:а|у|ом|е)?(?![а-яё])))(?!\d)",
    re.IGNORECASE,
)


class DatePrecision(Enum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


@dataclass(frozen=True, slots=True)
class PartialDate:
    """Дата с той точностью, с какой она названа: день, месяц и год или только год.

    Это период: «август 2024» — все дни с 1 по 31 августа 2024 года.
    """

    # Первый день периода.
    start: date
    precision: DatePrecision

    @classmethod
    def of_day(cls, day: date) -> PartialDate:
        return cls(day, DatePrecision.DAY)

    @classmethod
    def of_month(cls, year: int, month: int) -> PartialDate:
        return cls(date(year, month, 1), DatePrecision.MONTH)

    @classmethod
    def of_year(cls, year: int) -> PartialDate:
        return cls(date(year, 1, 1), DatePrecision.YEAR)

    @property
    def end(self) -> date:
        """Первый день после периода: полуинтервал, как у периода действия редакции."""
        return self.shifted(1).start

    def shifted(self, count: int) -> PartialDate:
        """Календарный период той же точности на `count` единиц позже или раньше."""
        if self.precision is DatePrecision.DAY:
            return PartialDate.of_day(self.start + timedelta(days=count))
        if self.precision is DatePrecision.YEAR:
            return PartialDate.of_year(self.start.year + count)
        months = self.start.year * 12 + self.start.month - 1 + count
        return PartialDate.of_month(months // 12, months % 12 + 1)


@dataclass(frozen=True, slots=True)
class DateMention:
    period: PartialDate
    # Место в тексте: по нему вызывающий видит слова перед датой и после неё.
    start: int
    end: int


def find_dates(text: str) -> tuple[DateMention, ...]:
    """Даты, названные в тексте полностью или с точностью до месяца либо года.

    «31.02.2025» и «31 февраля 2025» датами не считаются; год внутри них тоже.
    """
    mentions: list[DateMention] = []
    for match in _MENTION.finditer(text):
        period = _named_period(match)
        if period is not None:
            mentions.append(DateMention(period, match.start(), match.end()))
    return tuple(mentions)


def month_number(word: str) -> int:
    """Номер месяца по слову, найденному по MONTH_WORD_PATTERN."""
    lowered = word.lower()
    return next(
        number
        for number, forms in enumerate(_MONTH_WORDS, start=1)
        if re.fullmatch(forms, lowered)
    )


def calendar_period(year: int, month: int, day: str | None) -> PartialDate | None:
    """День месяца или месяц целиком, если день не назван. «31 февраля» — None."""
    if day is None:
        return PartialDate.of_month(year, month)
    found = _calendar_date(year, month, int(day))
    return None if found is None else PartialDate.of_day(found)


def _named_period(match: re.Match[str]) -> PartialDate | None:
    if match["numeric"] is not None:
        found = parse_date(match["numeric"])
        return None if found is None else PartialDate.of_day(found)
    if match["bare_year"] is not None:
        return PartialDate.of_year(int(match["bare_year"]))
    return calendar_period(
        int(match["year"]), month_number(match["month"]), match["day"]
    )


def parse_date(text: str) -> date | None:
    """Дата из фрагмента, найденного по DATE_PATTERN. «31.02.2025» — не дата."""
    numeric = _NUMERIC_DATE.fullmatch(text)
    if numeric is not None:
        day, month, year = (int(part) for part in numeric.groups())
        return _calendar_date(year, month, day)
    verbal = _VERBAL_DATE.fullmatch(text)
    if verbal is not None:
        month = _MONTHS.index(verbal.group(2).lower()) + 1
        return _calendar_date(int(verbal.group(3)), month, int(verbal.group(1)))
    return None


def _calendar_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
