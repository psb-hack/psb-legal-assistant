"""Дата события из вопроса: названная прямо или относительно дня вопроса.

День вопроса передаёт вызывающий: ядро не читает системное время, поэтому один и
тот же вопрос с одной и той же датой всегда читается одинаково.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from lex_dates import (
    MONTH_WORD_PATTERN,
    PartialDate,
    calendar_period,
    find_dates,
    month_number,
)

_SHIFT_BY_STEM = {
    "позапрошл": -2,
    "прошл": -1,
    "эт": 0,
    "текущ": 0,
    "нынешн": 0,
    "следующ": 1,
    "будущ": 1,
}
_DAY_SHIFT_BY_WORD = {"сегодня": 0, "вчера": -1, "позавчера": -2}
# «прошлом», «этого», «текущий»
_SHIFT = (
    rf"(?<![а-яё])(?P<shift>{'|'.join(_SHIFT_BY_STEM)})"
    r"(?:ый|ий|ого|его|ом|ем|ым|им|от)"
)
# «в августе прошлого года», «1 августа этого года»; без года — «в августе», «5 мая»
_MONTH_PHRASE = re.compile(
    rf"(?:(?<!\d)(?P<day>\d{{1,2}})\s+)?(?P<month>{MONTH_WORD_PATTERN})"
    rf"(?:\s+{_SHIFT}\s+года(?![а-яё]))?",
    re.IGNORECASE,
)
# «в прошлом году», «в этом месяце», «с начала текущего года»
_RELATIVE_PERIOD = re.compile(
    rf"{_SHIFT}\s+(?P<unit>год|месяц)(?:а|у|е|ом)?(?![а-яё])", re.IGNORECASE
)
# «сегодня», «вчера», «позавчера»
_RELATIVE_DAY = re.compile(
    rf"(?<![а-яё])(?P<word>{'|'.join(_DAY_SHIFT_BY_WORD)})(?![а-яё])", re.IGNORECASE
)
# «два года назад», «полгода назад», «3 месяца назад»: длительность, а не дата
_TIME_AGO = re.compile(
    r"(?:(?<![а-яё\d])(?:\d+|полтора|два|три|четыре|пять|несколько|пару)\s+)?"
    r"(?<![а-яё])"
    r"(?:год|года|лет|полгода|месяц[а-яё]{0,2}|недел[а-яё]{1,2}|день|дня|дней)"
    r"\s+назад(?![а-яё])",
    re.IGNORECASE,
)
# «от 12.02.2024»: так пишут дату документа — «договор № П-17/2024 от 12.02.2024»
_DOCUMENT_DATE_LEAD = re.compile(r"(?<![а-яё])от\s+$", re.IGNORECASE)


class DateRole(Enum):
    # «в августе 2024 года», «01.08.2024», «в прошлом году»
    STATED = "stated"
    # «от 12.02.2024»: реквизит документа или дата редакции
    DOCUMENT_DATE = "document_date"


@dataclass(frozen=True, slots=True)
class QuestionDate:
    period: PartialDate
    # Дословно, как в вопросе.
    wording: str
    role: DateRole


@dataclass(frozen=True, slots=True)
class EventDate:
    period: PartialDate
    wording: str


@dataclass(frozen=True, slots=True)
class UnclearEventDate:
    """По тексту не понять, какая дата — дата события, либо дату нельзя вычислить."""

    # Все даты вопроса в порядке текста.
    candidates: tuple[QuestionDate, ...]
    # Упоминания времени, из которых дата не получается: «в августе», «два года назад».
    incomplete: tuple[str, ...]


def read_event_date(question: str, today: date) -> EventDate | UnclearEventDate | None:
    """Дата события; None — в вопросе нет ни дат, ни упоминаний времени.

    Датой события считается единственная дата вопроса, если она названа не как
    реквизит документа. Между несколькими датами код не выбирает.
    """
    named, rest = _named_dates(question)
    relative, incomplete = _relative_dates(rest, today)
    candidates = tuple(found for _, found in sorted([*named, *relative], key=_position))
    if not candidates and not incomplete:
        return None
    periods = {found.period for found in candidates}
    stated = [found for found in candidates if found.role is DateRole.STATED]
    if len(periods) == 1 and stated and not incomplete:
        return EventDate(stated[0].period, stated[0].wording)
    return UnclearEventDate(candidates, incomplete)


_Found = tuple[int, QuestionDate]


def _named_dates(question: str) -> tuple[list[_Found], str]:
    """Даты с годом и текст без них: остаток читается как относительные даты."""
    mentions = find_dates(question)
    found: list[_Found] = []
    for mention in mentions:
        lead = _DOCUMENT_DATE_LEAD.search(question[: mention.start])
        role = DateRole.STATED if lead is None else DateRole.DOCUMENT_DATE
        wording = question[mention.start : mention.end]
        found.append((mention.start, QuestionDate(mention.period, wording, role)))
    return found, _blanked(question, ((item.start, item.end) for item in mentions))


def _relative_dates(text: str, today: date) -> tuple[list[_Found], tuple[str, ...]]:
    """Даты, отсчитанные от дня вопроса, и упоминания времени без даты."""
    found: list[_Found] = []
    incomplete: list[tuple[int, str]] = []
    month_phrases = list(_MONTH_PHRASE.finditer(text))
    for match in month_phrases:
        period = _month_of_relative_year(match, today)
        if period is None:
            incomplete.append((match.start(), match[0]))
        else:
            found.append(_stated(match, period))
    # «прошлого года» из «в августе прошлого года» — не отдельная дата.
    rest = _blanked(text, (match.span() for match in month_phrases))
    for match in _RELATIVE_PERIOD.finditer(rest):
        current = (
            PartialDate.of_year(today.year)
            if match["unit"].lower() == "год"
            else PartialDate.of_month(today.year, today.month)
        )
        shift = _SHIFT_BY_STEM[match["shift"].lower()]
        found.append(_stated(match, current.shifted(shift)))
    for match in _RELATIVE_DAY.finditer(rest):
        shift = _DAY_SHIFT_BY_WORD[match["word"].lower()]
        found.append(_stated(match, PartialDate.of_day(today).shifted(shift)))
    incomplete.extend((match.start(), match[0]) for match in _TIME_AGO.finditer(rest))
    return found, tuple(wording for _, wording in sorted(incomplete))


def _month_of_relative_year(match: re.Match[str], today: date) -> PartialDate | None:
    """«в августе прошлого года» → август; None — год не назван или дня такого нет."""
    if match["shift"] is None:
        return None
    year = today.year + _SHIFT_BY_STEM[match["shift"].lower()]
    return calendar_period(year, month_number(match["month"]), match["day"])


def _stated(match: re.Match[str], period: PartialDate) -> _Found:
    return match.start(), QuestionDate(period, match[0], DateRole.STATED)


def _position(found: _Found) -> int:
    return found[0]


def _blanked(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Текст той же длины, где уже прочитанные фрагменты заменены пробелами."""
    characters = list(text)
    for start, end in spans:
        characters[start:end] = " " * (end - start)
    return "".join(characters)
