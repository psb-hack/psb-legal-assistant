"""Шапка выгрузки статьи закона и пометки справочной системы об изменениях."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from lex_dates import DATE_PATTERN, parse_date
from lex_documents import AmendmentNote, LawAddress, LawUnit, LawUnitKind
from lex_law_lines import UNIT_NUMBER_PATTERN

# «Источник: Статья 70. ТК РФ Испытание при приеме на работу. Трудовой кодекс …»
_SOURCE_TITLE = re.compile(r"^Источник:\s*(?P<value>\S.*)$")
# «URL: https://base.garant.ru/12125268/»
_SOURCE_URL = re.compile(r"^URL:\s*(?P<value>\S+)$")
# «Дата выгрузки: 10.10.2026»
_RETRIEVED = re.compile(rf"^Дата\s+выгрузки:\s*(?P<value>{DATE_PATTERN})")
# «… (ТК РФ) (с изменениями и дополнениями)» → «ТК РФ»
_CODE_ABBREVIATION = re.compile(r"\((?P<short>[А-ЯЁ][А-Яа-яЁё]*\s+РФ)\)")
# «Федеральный закон от 27.07.2006 N 152-ФЗ "О персональных данных"» → «152-ФЗ»
_LAW_NUMBER = re.compile(r"[N№]\s*(?P<short>\d+-ФК?З)")
# «… (с изменениями и дополнениями)» в конце заголовка страницы
_REVISION_SUFFIX = re.compile(r"\s*\(с\s+изменениями[^)]*\)$")
# В заголовке страницы название акта идёт последним предложением.
_SENTENCE_BREAK = ". "

_KIND_BY_WORD = {
    "пункт": LawUnitKind.CLAUSE,
    "пунктом": LawUnitKind.CLAUSE,
    "подпункт": LawUnitKind.SUBCLAUSE,
    "подпунктом": LawUnitKind.SUBCLAUSE,
    "часть": LawUnitKind.PART,
    "частью": LawUnitKind.PART,
}
_ARTICLE_WORD = "статья"
_NOTES_WORD = "примечания"
# «Пункт 3 изменен с 1 сентября 2026 г. - Федеральный закон от … № 425-ФЗ»
# «Статья 13.11 дополнена частью 10 с 30 мая 2025 г. - Федеральный закон …»
# «Примечания изменены с 30 мая 2025 г. - Федеральный закон …»
_AMENDMENT = re.compile(
    rf"^(?P<subject>Статья|Пункт|Подпункт|Часть|Примечания)"
    rf"(?:\s+(?P<number>{UNIT_NUMBER_PATTERN}))?\s+"
    rf"(?:изменен[аы]?|дополнен[аы]?\s+(?P<added>подпунктом|пунктом|частью)"
    rf"\s+(?P<added_number>{UNIT_NUMBER_PATTERN}))"
    rf"\s+с\s+(?P<date>{DATE_PATTERN})\s+г\.(?:\s+[-–—]\s+(?P<act>\S.*))?$"
)


@dataclass(frozen=True, slots=True)
class LawSource:
    # «Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ (ТК РФ)»
    act: str
    act_short: str
    url: str
    retrieved: date | None


def read_source(lines: Iterable[str]) -> LawSource:
    """Акт, адрес страницы и дата выгрузки из строк шапки."""
    fields = list(lines)
    act = _REVISION_SUFFIX.sub(
        "", _first(_SOURCE_TITLE, fields).rsplit(_SENTENCE_BREAK, 1)[-1]
    )
    retrieved = _first(_RETRIEVED, fields)
    return LawSource(
        act=act,
        act_short=_act_short(act),
        url=_first(_SOURCE_URL, fields),
        retrieved=parse_date(retrieved) if retrieved else None,
    )


def read_amendment(line: str) -> AmendmentNote | None:
    """Пометка с адресом; адрес — единица так, как её назвала справочная система.

    Полный путь к единице («подп. 1.2» → «п. 1, подп. 1.2») знает только разбор статьи.
    """
    match = _AMENDMENT.match(line)
    if match is None:
        return None
    effective_from = parse_date(match["date"])
    if effective_from is None:
        return None
    return AmendmentNote(
        target=_target(match),
        effective_from=effective_from,
        source_act=match["act"],
        text=line,
    )


def _target(match: re.Match[str]) -> LawAddress | None:
    if match["added"] is not None:
        unit = LawUnit(_KIND_BY_WORD[match["added"]], match["added_number"])
        return LawAddress((unit,), None)
    subject = match["subject"].lower()
    if subject == _ARTICLE_WORD:
        return None
    if subject == _NOTES_WORD:
        return LawAddress((LawUnit(LawUnitKind.NOTE, None),), None)
    return LawAddress((LawUnit(_KIND_BY_WORD[subject], match["number"]),), None)


def _act_short(act: str) -> str:
    for pattern in (_CODE_ABBREVIATION, _LAW_NUMBER):
        match = pattern.search(act)
        if match is not None:
            return match["short"]
    return act


def _first(pattern: re.Pattern[str], lines: Iterable[str]) -> str:
    for line in lines:
        match = pattern.match(line)
        if match is not None:
            return match["value"]
    return ""
