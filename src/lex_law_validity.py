"""Действует ли текст нормы из базы на дату события.

Прежних редакций норм в базе нет: есть один текст на дату выгрузки и пометки
справочной системы об изменениях. По ним видно, новее ли текст, чем дата события.
Какой текст лежит в базе, если изменение названо, но на дату выгрузки ещё не
вступило в силу, из файла не следует.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from lex_dates import DATE_PATTERN, PartialDate, parse_date
from lex_documents import AmendmentNote, LawAddress, LawArticleHeader

# «Федеральным законом от 7 февраля 2017 г. N 13-ФЗ статья 13.11 настоящего Кодекса
# изложена в новой редакции, вступающей в силу с 1 июля 2017 г.»
# «… в новой редакции, распространяющейся на правоотношения, возникшие с 1 июля 2011 г.»
_EARLIER_CHANGE = re.compile(
    r"^Федеральным(?:и)?\s+закон(?:ом|ами)\s.*?"
    rf"(?:вступающ[а-яё]+\s+в\s+силу|возникшие)\s+с\s+(?P<date>{DATE_PATTERN})",
    re.IGNORECASE,
)


class Uncertainty(Enum):
    # Пометка называет изменение, которое на дату выгрузки ещё не вступило в силу.
    PENDING_AMENDMENT = "pending_amendment"
    # Период неполной даты захватывает день, с которого норма изменена.
    SPANS_AMENDMENT = "spans_amendment"
    # Дата события позже выгрузки: изменений после неё в базе нет.
    AFTER_RETRIEVAL = "after_retrieval"
    # Без даты выгрузки не отличить вступившее изменение от будущего.
    NO_RETRIEVAL_DATE = "no_retrieval_date"


@dataclass(frozen=True, slots=True)
class LawTextInForce:
    """Текст базы — редакция нормы, действовавшая на дату события."""

    # Последнее известное изменение нормы; None — пометок об изменении нет.
    amended: AmendmentNote | None
    # False — даты события в вопросе нет: текст приводится на дату выгрузки.
    date_stated: bool


@dataclass(frozen=True, slots=True)
class LawTextNewer:
    """Норма изменена позже даты события: редакции на эту дату в базе нет."""

    amended: AmendmentNote


@dataclass(frozen=True, slots=True)
class LawTextUncertain:
    reason: Uncertainty
    # Пометка, из-за которой исход не определён.
    note: AmendmentNote | None


LawTextOnDate = LawTextInForce | LawTextNewer | LawTextUncertain


def law_text_on(
    header: LawArticleHeader, address: LawAddress | None, period: PartialDate
) -> LawTextOnDate:
    """Исход для нормы по адресу; `address=None` — статья целиком."""
    if header.retrieved is None:
        return LawTextUncertain(Uncertainty.NO_RETRIEVAL_DATE, None)
    changes = _Changes.known(header, address, header.retrieved)
    latest = changes.latest
    if latest is not None and period.end <= latest.effective_from:
        return LawTextNewer(latest)
    if latest is not None and period.start < latest.effective_from:
        return LawTextUncertain(Uncertainty.SPANS_AMENDMENT, latest)
    if changes.pending is not None:
        return LawTextUncertain(Uncertainty.PENDING_AMENDMENT, changes.pending)
    if period.end > header.retrieved + timedelta(days=1):
        return LawTextUncertain(Uncertainty.AFTER_RETRIEVAL, None)
    return LawTextInForce(latest, date_stated=True)


def law_text_as_retrieved(
    header: LawArticleHeader, address: LawAddress | None
) -> LawTextInForce | LawTextUncertain:
    """Исход для вопроса без даты события: текст нормы на дату выгрузки."""
    if header.retrieved is None:
        return LawTextUncertain(Uncertainty.NO_RETRIEVAL_DATE, None)
    changes = _Changes.known(header, address, header.retrieved)
    if changes.pending is not None:
        return LawTextUncertain(Uncertainty.PENDING_AMENDMENT, changes.pending)
    return LawTextInForce(changes.latest, date_stated=False)


@dataclass(frozen=True, slots=True)
class _Changes:
    # Последнее изменение, вступившее в силу к дате выгрузки.
    latest: AmendmentNote | None
    # Ближайшее изменение, которое на дату выгрузки ещё не вступило в силу.
    pending: AmendmentNote | None

    @classmethod
    def known(
        cls, header: LawArticleHeader, address: LawAddress | None, retrieved: date
    ) -> _Changes:
        notes = [
            *(note for note in header.amendments if _concerns(note, address)),
            *_earlier_changes(header.comments),
        ]
        in_force = [note for note in notes if note.effective_from <= retrieved]
        pending = [note for note in notes if note.effective_from > retrieved]
        return cls(
            latest=max(in_force, key=_effective_from, default=None),
            pending=min(pending, key=_effective_from, default=None),
        )


def _concerns(note: AmendmentNote, address: LawAddress | None) -> bool:
    """Изменена сама единица, её родитель или статья целиком."""
    if note.target is None or address is None:
        return True
    changed = note.target.units
    # «Статья дополнена подпунктом 9.3» пути к единице не называет.
    if len(changed) == 1:
        return changed[0] in address.units
    return address.units[: len(changed)] == changed


def _earlier_changes(comments: tuple[str, ...]) -> list[AmendmentNote]:
    """Изменения, записанные справочной системой комментарием, а не пометкой.

    Единица в таком комментарии названа словами и здесь не разбирается: изменение
    относится ко всей статье. Для нормы, которой оно не касалось, это даёт
    «редакции на дату нет» вместо «действует» — ошибка в безопасную сторону.
    """
    changes: list[AmendmentNote] = []
    for comment in comments:
        match = _EARLIER_CHANGE.match(comment)
        effective_from = None if match is None else parse_date(match["date"])
        if effective_from is not None:
            changes.append(AmendmentNote(None, effective_from, None, comment))
    return changes


def _effective_from(note: AmendmentNote) -> date:
    return note.effective_from
