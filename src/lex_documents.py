"""Доменные типы разобранного документа и ссылка на положение."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import NewType

CLAUSE_ABBREVIATION = "п."
PARAGRAPH_ABBREVIATION = "абз."
PREAMBLE_LABEL = "преамбула"
SECTION_LABEL = "раздел"
EDITION_DATE_FORMAT = "%d.%m.%Y"

DocumentId = NewType("DocumentId", str)
ProvisionId = NewType("ProvisionId", str)


class DocumentPart(Enum):
    """Место положения в структуре документа. Смысл текста здесь не оценивается."""

    PREAMBLE = "preamble"
    BODY = "body"
    REQUISITES = "requisites"


@dataclass(frozen=True, slots=True)
class ClauseNumber:
    parts: tuple[int, ...]

    def __str__(self) -> str:
        return ".".join(str(part) for part in self.parts)

    @property
    def label(self) -> str:
        return f"{CLAUSE_ABBREVIATION} {self}"

    def is_ancestor_of(self, other: ClauseNumber) -> bool:
        depth = len(self.parts)
        return depth < len(other.parts) and other.parts[:depth] == self.parts


@dataclass(frozen=True, slots=True)
class ParagraphNumber:
    """Номер абзаца основного текста в документе без нумерованных пунктов."""

    position: int

    @property
    def label(self) -> str:
        return f"{PARAGRAPH_ABBREVIATION} {self.position}"


@dataclass(frozen=True, slots=True)
class Heading:
    number: ClauseNumber
    title: str


@dataclass(frozen=True, slots=True)
class Provision:
    # Порядковый номер в документе: по нему строится идентификатор положения.
    position: int
    part: DocumentPart
    number: ClauseNumber | ParagraphNumber | None
    # От раздела до ближайшего родителя: без вводной фразы подпункт теряет смысл.
    headings: tuple[Heading, ...]
    text: str

    @property
    def label(self) -> str:
        return PREAMBLE_LABEL if self.number is None else self.number.label


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    provisions: tuple[Provision, ...]


class DocumentKind(Enum):
    CONTRACT = "contract"
    SUPPLEMENTARY_AGREEMENT = "supplementary_agreement"
    REGULATION = "regulation"
    LETTER = "letter"
    MEMO = "memo"
    UNKNOWN = "unknown"


class DocumentStatus(Enum):
    """Статус по пометке в шапке. Срок действия из текста договора сюда не входит."""

    IN_FORCE = "in_force"
    REPEALED = "repealed"


@dataclass(frozen=True, slots=True)
class Party:
    name: str
    # ИНН различает контрагентов с похожими наименованиями.
    taxpayer_number: str | None
    role: str


@dataclass(frozen=True, slots=True)
class DocumentHeader:
    kind: DocumentKind
    # Пустая строка, если шапки в тексте нет.
    title: str
    number: str | None
    dated: date | None
    parties: tuple[Party, ...]
    edition: date | None
    effective_from: date | None
    status: DocumentStatus
    repealed_from: date | None

    @property
    def citation_title(self) -> str:
        if self.edition is None:
            return self.title
        return f"{self.title} в редакции от {self.edition:{EDITION_DATE_FORMAT}}"


@dataclass(frozen=True, slots=True)
class Document:
    id: DocumentId
    header: DocumentHeader
    provisions: tuple[Provision, ...]

    def provision_id(self, provision: Provision) -> ProvisionId:
        return ProvisionId(f"{self.id}.{provision.position}")


def cite(document: Document, provision: Provision) -> str:
    return f"{document.header.citation_title}, {provision.label}"


def context_heading(document: Document, provision: Provision) -> str:
    """Заголовок фрагмента для индекса: документ, стороны, путь разделов, адрес."""
    title = document.header.citation_title
    parties = ", ".join(party.name for party in document.header.parties)
    path = [f"{title} ({parties})" if parties else title]
    for depth, heading in enumerate(provision.headings):
        unit = SECTION_LABEL if depth == 0 else CLAUSE_ABBREVIATION
        path.append(f"{unit} {heading.number} «{heading.title}»")
    path.append(provision.label)
    return ", ".join(path)
