"""Доменные типы разобранного документа и ссылка на положение."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

CLAUSE_ABBREVIATION = "п."
PREAMBLE_LABEL = "преамбула"


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


def cite(document_title: str, provision: Provision) -> str:
    return f"{document_title}, {provision.label}"
