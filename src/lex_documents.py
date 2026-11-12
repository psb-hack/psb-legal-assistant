"""Доменные типы разобранного документа и ссылка на положение."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

CLAUSE_ABBREVIATION = "п."
PREAMBLE_LABEL = "преамбула"
EDITION_DATE_FORMAT = "%d.%m.%Y"


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

    def is_ancestor_of(self, other: ClauseNumber) -> bool:
        depth = len(self.parts)
        return depth < len(other.parts) and other.parts[:depth] == self.parts


@dataclass(frozen=True, slots=True)
class Heading:
    number: ClauseNumber
    title: str


@dataclass(frozen=True, slots=True)
class Provision:
    part: DocumentPart
    number: ClauseNumber | None
    # От раздела до ближайшего родителя: без вводной фразы подпункт теряет смысл.
    headings: tuple[Heading, ...]
    text: str

    @property
    def label(self) -> str:
        if self.number is None:
            return PREAMBLE_LABEL
        return f"{CLAUSE_ABBREVIATION} {self.number}"


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


def cite(document_title: str, provision: Provision) -> str:
    return f"{document_title}, {provision.label}"
