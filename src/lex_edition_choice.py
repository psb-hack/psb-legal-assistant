"""Редакция документа компании на дату события.

Граф отвечает, какая редакция действовала в один день. Здесь решается, что делать
с датой, названной до месяца или года, и с вопросом без даты.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from lex_dates import PartialDate
from lex_document_graph import Edition, EditionFamily
from lex_documents import DocumentHeader, DocumentStatus
from lex_knowledge_base import KnowledgeBase


class MissingEdition(Enum):
    # Дата раньше начала действия первой редакции из базы.
    BEFORE_FIRST_EDITION = "before_first_edition"
    # К этой дате редакция утратила силу, а следующей в базе нет.
    NOT_IN_FORCE = "not_in_force"


@dataclass(frozen=True, slots=True)
class EditionInForce:
    edition: Edition
    # False — даты события в вопросе нет: взята действующая редакция, и ответ
    # должен это оговорить.
    date_stated: bool


@dataclass(frozen=True, slots=True)
class EditionsAcrossBoundary:
    """Период захватывает смену редакции: «в 2025 году» при смене 01.03.2025."""

    # Редакции, действовавшие в периоде, по порядку.
    editions: tuple[Edition, ...]
    # Дни внутри периода, с которых действует уже другая редакция или никакая.
    boundaries: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class NoEditionOnDate:
    reason: MissingEdition


@dataclass(frozen=True, slots=True)
class RepealDateUnknown:
    """Редакция помечена утратившей силу без даты: действовала ли она в названный
    период, по базе не установить."""

    edition: Edition


EditionChoice = (
    EditionInForce | EditionsAcrossBoundary | NoEditionOnDate | RepealDateUnknown
)


def edition_on(
    family: EditionFamily, period: PartialDate, base: KnowledgeBase
) -> EditionChoice:
    """Редакция, действовавшая весь названный период, или причина её не выбирать."""
    days = [
        period.start,
        *(day for day in _change_days(family) if period.start < day < period.end),
    ]
    in_force = [family.edition_at(day) for day in days]
    boundaries = tuple(
        day
        for before, after, day in zip(in_force, in_force[1:], days[1:], strict=False)
        if before != after
    )
    if boundaries:
        editions = dict.fromkeys(item for item in in_force if item is not None)
        return EditionsAcrossBoundary(tuple(editions), boundaries)
    edition = in_force[0]
    if edition is None:
        return NoEditionOnDate(_missing(family, period.start))
    if _repeal_date_unknown(edition, base):
        return RepealDateUnknown(edition)
    return EditionInForce(edition, date_stated=True)


def current_edition(
    family: EditionFamily, today: date, base: KnowledgeBase
) -> EditionInForce | NoEditionOnDate:
    """Действующая редакция — для вопроса, в котором даты события нет."""
    edition = family.edition_at(today)
    # «Утратила силу» без даты: когда — неизвестно, но к дню вопроса уже утратила.
    if edition is None or _repeal_date_unknown(edition, base):
        return NoEditionOnDate(_missing(family, today))
    return EditionInForce(edition, date_stated=False)


def _change_days(family: EditionFamily) -> list[date]:
    """Дни, с которых может действовать другая редакция, по возрастанию."""
    days = {edition.effective_from for edition in family.editions}
    days.update(
        edition.effective_to
        for edition in family.editions
        if edition.effective_to is not None
    )
    return sorted(days)


def _missing(family: EditionFamily, on: date) -> MissingEdition:
    if on < family.editions[0].effective_from:
        return MissingEdition.BEFORE_FIRST_EDITION
    return MissingEdition.NOT_IN_FORCE


def _repeal_date_unknown(edition: Edition, base: KnowledgeBase) -> bool:
    if edition.effective_to is not None:
        return False
    return any(
        document.id == edition.document
        and isinstance(document.header, DocumentHeader)
        and document.header.status is DocumentStatus.REPEALED
        for document in base.documents
    )
