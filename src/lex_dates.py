"""Даты в деловых текстах: «01.03.2025» и «1 марта 2025 г.»."""

from __future__ import annotations

import re
from datetime import date

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

# «01.03.2025»
_NUMERIC_DATE = re.compile(rf"({_DAY})\.({_MONTH_NUMBER})\.({_YEAR})")
# «1 марта 2025»
_VERBAL_DATE = re.compile(rf"({_DAY})\s+({_MONTH_NAME})\s+({_YEAR})", re.IGNORECASE)

# Без групп: шаблон вставляют в выражения других модулей, иногда дважды в одно.
DATE_PATTERN = (
    rf"(?<!\d)(?:{_DAY}\.{_MONTH_NUMBER}\.{_YEAR}"
    rf"|{_DAY}\s+{_MONTH_NAME}\s+{_YEAR})(?!\d)"
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
