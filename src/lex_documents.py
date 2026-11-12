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
ARTICLE_ABBREVIATION = "ст."
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


class LawUnitKind(Enum):
    """Единица статьи закона; значение — сокращение, принятое в ссылках."""

    PART = "ч."
    CLAUSE = "п."
    SUBCLAUSE = "подп."
    NOTE = "прим."


@dataclass(frozen=True, slots=True)
class LawUnit:
    kind: LawUnitKind
    # Строка, а не число: в кодексах есть «1.8-3». None — единица без номера,
    # например раздел «Примечания» целиком.
    number: str | None

    @property
    def label(self) -> str:
        if self.number is None:
            return self.kind.value
        return f"{self.kind.value} {self.number}"


@dataclass(frozen=True, slots=True)
class ParagraphRange:
    first: int
    last: int

    @property
    def label(self) -> str:
        if self.first == self.last:
            return f"{PARAGRAPH_ABBREVIATION} {self.first}"
        return f"{PARAGRAPH_ABBREVIATION} {self.first}–{self.last}"


@dataclass(frozen=True, slots=True)
class LawAddress:
    """Путь к норме внутри статьи: «ч. 3, п. 2» или «п. 1, подп. 2.1, абз. 4–6»."""

    units: tuple[LawUnit, ...]
    # Заполнен, когда на единицу приходится несколько положений или когда адрес
    # указывает на абзацы, утратившие силу.
    paragraphs: ParagraphRange | None

    @property
    def label(self) -> str:
        steps = [unit.label for unit in self.units]
        if self.paragraphs is not None:
            steps.append(self.paragraphs.label)
        return ", ".join(steps)


@dataclass(frozen=True, slots=True)
class Heading:
    number: ClauseNumber | LawUnit
    title: str


@dataclass(frozen=True, slots=True)
class Provision:
    # Порядковый номер в документе: по нему строится идентификатор положения.
    position: int
    part: DocumentPart
    number: ClauseNumber | ParagraphNumber | LawAddress | None
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

    @property
    def context_title(self) -> str:
        parties = ", ".join(party.name for party in self.parties)
        if not parties:
            return self.citation_title
        return f"{self.citation_title} ({parties})"


@dataclass(frozen=True, slots=True)
class AmendmentNote:
    """Пометка справочной системы об изменении статьи или её единицы."""

    # None — статья целиком.
    target: LawAddress | None
    effective_from: date
    # «Федеральный закон от 28 ноября 2025 г. № 425-ФЗ»; в пометке бывает несколько.
    source_act: str | None
    text: str


@dataclass(frozen=True, slots=True)
class RepealedUnit:
    """Строка на месте единицы, утратившей силу: занимает адрес, нормы не содержит."""

    address: LawAddress
    text: str


@dataclass(frozen=True, slots=True)
class LawArticleHeader:
    # «Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ (ТК РФ)»
    act: str
    # «ТК РФ», «152-ФЗ» — так акт называют в ссылках.
    act_short: str
    article: str
    # Пустая строка, если после номера статьи сразу идёт норма.
    title: str
    source_url: str
    retrieved: date | None
    amendments: tuple[AmendmentNote, ...]
    # Комментарии справочной системы без адреса: судебная практика, переходные
    # положения, отсылки к формам. Нормами не являются.
    comments: tuple[str, ...]
    repealed: tuple[RepealedUnit, ...]

    @property
    def citation_title(self) -> str:
        return f"{self.act_short}, {ARTICLE_ABBREVIATION} {self.article}"

    @property
    def context_title(self) -> str:
        if not self.title:
            return self.citation_title
        return f"{self.citation_title} «{self.title}»"


@dataclass(frozen=True, slots=True)
class Document:
    id: DocumentId
    header: DocumentHeader | LawArticleHeader
    provisions: tuple[Provision, ...]

    def provision_id(self, provision: Provision) -> ProvisionId:
        return ProvisionId(f"{self.id}.{provision.position}")


def cite(document: Document, provision: Provision) -> str:
    return f"{document.header.citation_title}, {provision.label}"


def context_heading(document: Document, provision: Provision) -> str:
    """Заголовок фрагмента для индекса: документ, стороны, путь разделов, адрес."""
    path = [document.header.context_title]
    for depth, heading in enumerate(provision.headings):
        path.append(f"{_heading_label(depth, heading.number)} «{heading.title}»")
    path.append(provision.label)
    return ", ".join(path)


def _heading_label(depth: int, number: ClauseNumber | LawUnit) -> str:
    if isinstance(number, LawUnit):
        return number.label
    unit = SECTION_LABEL if depth == 0 else CLAUSE_ABBREVIATION
    return f"{unit} {number}"
