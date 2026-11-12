"""Каталог сущностей базы: контрагенты и документы компании, собранные из шапок.

Каталог строится из разобранных документов. Имена файлов и заранее известные
наименования в нём не участвуют: на другой базе он соберётся из её шапок.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

from lex_declension import case_forms
from lex_document_header import LEGAL_FORMS
from lex_document_numbers import DocumentNumber, parse_document_number
from lex_documents import Document, DocumentHeader, DocumentId, DocumentKind, Party
from lex_knowledge_base import KnowledgeBase
from lex_normalization import comparison_key

# В ключе сравнения все кавычки прямые: «ооо "альфа-трейд"» → «альфа-трейд».
_QUOTED_NAME = re.compile(r'"([^"]+)"')
# «альфа-трейд» → «альфа», «трейд»: дефис и пробел в наименовании равнозначны.
_NAME_WORD = re.compile(r"[^\W_]+")
# «склад» в «склад № 3», но не «Склад» и не «СЭД»: слово целиком строчными.
_LOWERCASE_WORD = re.compile(r"(?<![^\W_])[а-яёa-z]+(?![^\W_])")
_LEGAL_FORM_KEYS = frozenset(comparison_key(form) for form in LEGAL_FORMS)


@dataclass(frozen=True, slots=True)
class NameToken:
    """Слово наименования контрагента и его падежные формы в ключе сравнения."""

    nominative: str
    forms: frozenset[str]
    # Какая-то форма слова встречается в текстах базы со строчной буквы: «склад»
    # в «Северный Склад». Одно такое слово в вопросе контрагента не называет.
    common: bool


@dataclass(frozen=True, slots=True)
class Counterparty:
    # Как в шапке: «ООО «Альфа-Трейд»».
    name: str
    taxpayer_number: str | None
    # У одного контрагента в разных договорах роли разные.
    roles: tuple[str, ...]
    documents: tuple[DocumentId, ...]
    # Наименование без организационно-правовой формы и кавычек.
    name_tokens: tuple[NameToken, ...]


@dataclass(frozen=True, slots=True)
class DocumentEntry:
    id: DocumentId
    kind: DocumentKind
    title: str
    # None — номера нет или он не вида «буквы, число, год»: «№ 1» допсоглашения.
    number: DocumentNumber | None
    edition: date | None


@dataclass(frozen=True, slots=True)
class EntityCatalog:
    counterparties: tuple[Counterparty, ...]
    documents: tuple[DocumentEntry, ...]


def build_catalog(base: KnowledgeBase) -> EntityCatalog:
    """Контрагенты и документы компании; статьи законов в каталог не входят."""
    company_documents = [
        (document.id, document.header)
        for document in base.documents
        if isinstance(document.header, DocumentHeader)
    ]
    lowercase_words = _lowercase_words(base.documents)
    return EntityCatalog(
        counterparties=tuple(_counterparties(company_documents, lowercase_words)),
        documents=tuple(
            DocumentEntry(
                id=document_id,
                kind=header.kind,
                title=header.citation_title,
                number=(
                    None
                    if header.number is None
                    else parse_document_number(header.number)
                ),
                edition=header.edition,
            )
            for document_id, header in company_documents
        ),
    )


def _lowercase_words(documents: Iterable[Document]) -> frozenset[str]:
    return frozenset(
        comparison_key(word)
        for document in documents
        for provision in document.provisions
        for word in _LOWERCASE_WORD.findall(provision.text)
    )


def _counterparties(
    documents: list[tuple[DocumentId, DocumentHeader]], lowercase_words: frozenset[str]
) -> Iterator[Counterparty]:
    appearances: dict[str, list[tuple[DocumentId, Party]]] = {}
    for document_id, header in documents:
        for party in header.parties:
            key = comparison_key(party.name)
            appearances.setdefault(key, []).append((document_id, party))
    for same_name in appearances.values():
        for taxpayer_number, own in _split_by_taxpayer_number(same_name):
            name = own[0][1].name
            yield Counterparty(
                name=name,
                taxpayer_number=taxpayer_number,
                roles=tuple(dict.fromkeys(party.role for _, party in own)),
                documents=tuple(dict.fromkeys(document_id for document_id, _ in own)),
                name_tokens=tuple(
                    _name_token(word, lowercase_words) for word in _name_words(name)
                ),
            )


def _split_by_taxpayer_number(
    same_name: list[tuple[DocumentId, Party]],
) -> list[tuple[str | None, list[tuple[DocumentId, Party]]]]:
    """Одно наименование — один контрагент, пока у него не больше одного ИНН.

    В допсоглашении сторона названа без ИНН: это тот же контрагент, что в
    договоре. Если ИНН у наименования несколько, это разные лица, и сторону
    без ИНН нельзя отнести ни к одному из них — она остаётся отдельной записью.
    """
    numbers = list(dict.fromkeys(party.taxpayer_number for _, party in same_name))
    known = [number for number in numbers if number is not None]
    if len(known) <= 1:
        return [(known[0] if known else None, same_name)]
    return [
        (
            number,
            [
                appearance
                for appearance in same_name
                if appearance[1].taxpayer_number == number
            ],
        )
        for number in numbers
    ]


def _name_words(name: str) -> list[str]:
    key = comparison_key(name)
    quoted = _QUOTED_NAME.search(key)
    if quoted is not None:
        return _NAME_WORD.findall(quoted.group(1))
    words: list[str] = _NAME_WORD.findall(key)
    while len(words) > 1 and words[0] in _LEGAL_FORM_KEYS:
        words = words[1:]
    return words


def _name_token(word: str, lowercase_words: frozenset[str]) -> NameToken:
    forms = case_forms(word)
    return NameToken(
        nominative=word, forms=forms, common=not forms.isdisjoint(lowercase_words)
    )
