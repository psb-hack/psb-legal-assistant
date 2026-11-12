"""Граф связей между документами компании: допсоглашения, редакции, дополнения.

Граф строится из текста документов и не меняется. Связь, адресата которой в базе
нет, остаётся в графе неразрешённой: по ней ответ может сказать, чего не хватает.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NewType

from lex_declension import case_forms
from lex_document_links import (
    DocumentReference,
    LinkKind,
    LinkStatement,
    find_link_statements,
    title_words,
)
from lex_document_numbers import parse_document_number
from lex_documents import (
    ClauseNumber,
    Document,
    DocumentHeader,
    DocumentId,
    ProvisionId,
)
from lex_knowledge_base import KnowledgeBase
from lex_normalization import comparison_key

FamilyKey = NewType("FamilyKey", str)

# «п. 3 Положения» подходит к любому положению базы; два слова названия подряд —
# наименьшее совпадение, по которому документ считается названным.
_MIN_TITLE_WORDS = 2


@dataclass(frozen=True, slots=True)
class Link:
    kind: LinkKind
    source: DocumentId
    # Положение источника, в котором связь названа.
    stated_in: ProvisionId
    # Адресат дословно, как в источнике.
    reference: str
    # None — документа-адресата в базе нет или под запись подходит несколько.
    target: DocumentId | None
    # Пункт адресата, названный в источнике, и его положение, если оно нашлось.
    clause: ClauseNumber | None
    target_provision: ProvisionId | None
    effective_from: date | None

    @property
    def is_resolved(self) -> bool:
        if self.target is None:
            return False
        return self.clause is None or self.target_provision is not None


@dataclass(frozen=True, slots=True)
class Edition:
    document: DocumentId
    dated: date
    effective_from: date
    # Первый день, когда редакция уже не действует. None — действует.
    effective_to: date | None

    def in_force(self, on: date) -> bool:
        if on < self.effective_from:
            return False
        return self.effective_to is None or on < self.effective_to


@dataclass(frozen=True, slots=True)
class EditionFamily:
    """Редакции одного документа, лежащие в базе разными файлами."""

    key: FamilyKey
    title: str
    # По возрастанию начала действия.
    editions: tuple[Edition, ...]

    def edition_at(self, on: date) -> Edition | None:
        """Редакция, действовавшая на дату; None — такой редакции в базе нет."""
        # Периоды пересекаются, только если тексты редакций противоречат друг
        # другу; тогда берётся более поздняя редакция.
        for edition in reversed(self.editions):
            if edition.in_force(on):
                return edition
        return None


@dataclass(frozen=True, slots=True)
class DocumentGraph:
    links: tuple[Link, ...]
    families: tuple[EditionFamily, ...]

    def links_from(self, document: DocumentId) -> tuple[Link, ...]:
        return tuple(link for link in self.links if link.source == document)

    def links_to(self, document: DocumentId) -> tuple[Link, ...]:
        return tuple(link for link in self.links if link.target == document)

    def amendments_of(self, document: DocumentId) -> tuple[DocumentId, ...]:
        """Допсоглашения к документу в порядке документов базы."""
        return tuple(
            link.source
            for link in self.links_to(document)
            if link.kind is LinkKind.AMENDS
        )

    def family_of(self, document: DocumentId) -> EditionFamily | None:
        """Семейство редакций документа; None — у документа редакций нет."""
        for family in self.families:
            if any(edition.document == document for edition in family.editions):
                return family
        return None

    @property
    def unresolved(self) -> tuple[Link, ...]:
        return tuple(link for link in self.links if not link.is_resolved)


@dataclass(frozen=True, slots=True)
class _CompanyDocument:
    document: Document
    header: DocumentHeader
    title_words: tuple[str, ...]


def build_graph(base: KnowledgeBase) -> DocumentGraph:
    """Связи и редакции документов компании; статьи законов в граф не входят."""
    company = [
        _CompanyDocument(document, document.header, title_words(document.header.title))
        for document in base.documents
        if isinstance(document.header, DocumentHeader)
    ]
    families = _families(company)
    return DocumentGraph(
        links=tuple(
            _link(source, statement, company, families)
            for source in company
            for statement in find_link_statements(source.document)
        ),
        families=families,
    )


def _families(company: list[_CompanyDocument]) -> tuple[EditionFamily, ...]:
    members: dict[FamilyKey, list[tuple[date, _CompanyDocument]]] = {}
    for item in company:
        if item.header.edition is not None:
            members.setdefault(_family_key(item.header), []).append(
                (item.header.edition, item)
            )
    return tuple(
        EditionFamily(key, same[0][1].header.title, _editions(same))
        for key, same in members.items()
    )


def _family_key(header: DocumentHeader) -> FamilyKey:
    return FamilyKey(f"{header.kind.value}:{comparison_key(header.title)}")


def _editions(members: list[tuple[date, _CompanyDocument]]) -> tuple[Edition, ...]:
    # Без «Действует с …» в шапке редакция действует с даты редакции.
    ordered = sorted(
        ((member.header.effective_from or dated, dated, member))
        for dated, member in members
    )
    editions: list[Edition] = []
    for index, (effective_from, dated, member) in enumerate(ordered):
        following = ordered[index + 1][0] if index + 1 < len(ordered) else None
        editions.append(
            Edition(
                document=member.document.id,
                dated=dated,
                effective_from=effective_from,
                # Пометка об отмене в самой редакции точнее, чем начало следующей.
                effective_to=member.header.repealed_from or following,
            )
        )
    return tuple(editions)


def _link(
    source: _CompanyDocument,
    statement: LinkStatement,
    company: list[_CompanyDocument],
    families: tuple[EditionFamily, ...],
) -> Link:
    target = _resolve(source, statement.reference, company, families)
    clause = statement.clause
    return Link(
        kind=statement.kind,
        source=source.document.id,
        stated_in=source.document.provision_id(statement.stated_in),
        reference=statement.reference.text,
        target=None if target is None else target.document.id,
        clause=clause,
        target_provision=(
            None
            if target is None or clause is None
            else _provision_at(target.document, clause)
        ),
        effective_from=statement.effective_from,
    )


def _resolve(
    source: _CompanyDocument,
    reference: DocumentReference,
    company: list[_CompanyDocument],
    families: tuple[EditionFamily, ...],
) -> _CompanyDocument | None:
    others = [item for item in company if item is not source]
    if reference.number is not None:
        candidates = [
            item
            for item in others
            if item.header.number is not None
            and _same_number(reference.number, item.header.number)
        ]
    elif reference.title_words:
        candidates = _named_by_title(reference.title_words, others)
    else:
        own_family = _family_key(source.header)
        candidates = [item for item in others if _family_key(item.header) == own_family]
    if reference.edition is not None:
        candidates = [
            item for item in candidates if item.header.edition == reference.edition
        ]
    elif len(candidates) > 1 and source.header.dated is not None:
        candidates = _in_force(candidates, source.header.dated, families)
    return candidates[0] if len(candidates) == 1 else None


def _same_number(written: str, own: str) -> bool:
    written_parts = parse_document_number(written)
    own_parts = parse_document_number(own)
    if written_parts is None or own_parts is None:
        # «№ 14-П» на части не разбирается: сравнивается запись целиком.
        return comparison_key(written) == comparison_key(own)
    return written_parts.same_series(own_parts) and written_parts.year == own_parts.year


def _named_by_title(
    written: tuple[str, ...], documents: list[_CompanyDocument]
) -> list[_CompanyDocument]:
    """Документы, название которых начинается с наибольшего числа названных слов."""
    shared = [_shared_words(written, item.title_words) for item in documents]
    longest = max(shared, default=0)
    if longest < _MIN_TITLE_WORDS:
        return []
    return [
        item
        for item, length in zip(documents, shared, strict=True)
        if length == longest
    ]


def _shared_words(written: tuple[str, ...], own: tuple[str, ...]) -> int:
    """Сколько первых слов названия совпало; источник пишет их в любом падеже."""
    length = 0
    for written_word, own_word in zip(written, own, strict=False):
        if written_word != own_word and written_word not in case_forms(own_word):
            break
        length += 1
    return length


def _in_force(
    candidates: list[_CompanyDocument], on: date, families: tuple[EditionFamily, ...]
) -> list[_CompanyDocument]:
    """Ссылка без даты редакции называет редакцию, действовавшую на дату источника."""
    named = {item.document.id for item in candidates}
    for family in families:
        if named <= {edition.document for edition in family.editions}:
            edition = family.edition_at(on)
            return [
                item
                for item in candidates
                if edition is not None and item.document.id == edition.document
            ]
    return candidates


def _provision_at(document: Document, clause: ClauseNumber) -> ProvisionId | None:
    for provision in document.provisions:
        if provision.number == clause:
            return document.provision_id(provision)
    return None
