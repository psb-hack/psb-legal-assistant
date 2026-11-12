"""Утверждения о связях между документами, найденные в тексте одного документа.

Связь здесь ещё не разрешена: адресат записан так, как его называет источник.
Какой документ базы имеется в виду, решает граф (lex_document_graph).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from lex_dates import DATE_PATTERN, parse_date
from lex_document_header import REPEALED_EDITION
from lex_documents import (
    ClauseNumber,
    Document,
    DocumentHeader,
    DocumentKind,
    DocumentPart,
    Provision,
)
from lex_normalization import comparison_key

_NUMBER_SIGN = "№"
_WORD = r"[^\W\d_]+"
# «п. 5.4», «пункт 3.2», «пунктом 12»
_CLAUSE = r"(?P<address>(?:пункт\w*|п\.)\s*(?P<clause>\d+(?:\.\d+)*))"
# «Регламента согласования договоров»: от прописной буквы до скобки, знака номера,
# знака препинания или слов «в редакции».
_TITLE = (
    rf"(?P<title>[А-ЯЁ]{_WORD}(?:\s+{_WORD})*?)"
    rf"(?=\s*(?:[(,.;:{_NUMBER_SIGN}]|$)|\s+в\s+редакции)"
)
# « № П-17/2024», « № 14-П»
_NUMBER = rf"(?:\s*{_NUMBER_SIGN}\s*(?P<number>[^\s,;()]+))?"
# « (ред. от 01.03.2025)», « в редакции от 1 марта 2025 г.»
_EDITION = (
    rf"(?:\s*\(?(?:в\s+)?ред(?:акци\w+|\.)\s*от\s+(?P<edition>{DATE_PATTERN})"
    r"(?:\s*г\.)?\)?)?"
)
# «к договору поставки № П-17/2024 от 12 февраля 2024 г.» — вторая строка названия
# допсоглашения.
_PARENT_DOCUMENT = re.compile(
    rf"^к\s+(?P<title>{_WORD}(?:\s+{_WORD})*){_NUMBER}", re.IGNORECASE
)
# «в дополнение к порядку, установленному п. 5.4 Регламента согласования договоров
# (ред. от 01.03.2025)»; пункт может быть не назван: «в дополнение к Регламенту …».
_SUPPLEMENT = re.compile(
    rf"[Вв]\s+дополнение\s+к\s+(?:[^.;:]*?{_CLAUSE}\s+)?{_TITLE}{_NUMBER}{_EDITION}"
)
# «Настоящее дополнительное соглашение вступает в силу с даты подписания»
_FROM_SIGNING = re.compile(
    r"вступает\s+в\s+силу\s+с\s+(?:даты|момента)\s+(?:его\s+)?подписания",
    re.IGNORECASE,
)
# «Соглашение вступает в силу с 1 января 2031 г.», «Порядок действует с 01.09.2031»
_COMMENCEMENT_DATE = re.compile(
    r"(?:вступает\s+в\s+силу|действует|вводится\s+в\s+действие|применяется)"
    rf"\s+с\s+(?P<date>{DATE_PATTERN})",
    re.IGNORECASE,
)
# «С 21 апреля 2025 г. до подписания любого договора … инициатор обязан …»
_LEADING_DATE = re.compile(rf"^С\s+(?P<date>{DATE_PATTERN})")
_TITLE_WORD = re.compile(r"[^\W_]+")


class LinkKind(Enum):
    # Допсоглашение к договору.
    AMENDS = "amends"
    # Новая редакция документа отменяет прежнюю.
    SUPERSEDES = "supersedes"
    # Требование добавлено к положению другого документа, текст положения прежний.
    SUPPLEMENTS = "supplements"


@dataclass(frozen=True, slots=True)
class DocumentReference:
    """Адресат связи так, как он назван в тексте источника."""

    # Дословно: «п. 5.4 Регламента согласования договоров (ред. от 01.03.2025)».
    text: str
    # Слова названия в ключе сравнения и в том падеже, в каком они записаны.
    # Пусто — названа только редакция, речь о другой редакции самого источника.
    title_words: tuple[str, ...]
    number: str | None
    edition: date | None


@dataclass(frozen=True, slots=True)
class LinkStatement:
    kind: LinkKind
    # Положение источника, в котором связь названа.
    stated_in: Provision
    reference: DocumentReference
    # Пункт адресата, если он назван.
    clause: ClauseNumber | None
    effective_from: date | None


def find_link_statements(document: Document) -> tuple[LinkStatement, ...]:
    """Связи, названные в тексте документа компании, в порядке положений."""
    header = document.header
    if not isinstance(header, DocumentHeader):
        return ()
    effective_from = _commencement(document.provisions, header)
    return tuple(
        LinkStatement(kind, provision, reference, clause, effective_from)
        for provision in document.provisions
        for kind, reference, clause in _named_targets(provision, header)
    )


def title_words(title: str) -> tuple[str, ...]:
    """«Регламента согласования договоров» → («регламента», «согласования», …)."""
    return tuple(_TITLE_WORD.findall(comparison_key(title)))


def _named_targets(
    provision: Provision, header: DocumentHeader
) -> Iterator[tuple[LinkKind, DocumentReference, ClauseNumber | None]]:
    is_title_line = provision.part is DocumentPart.PREAMBLE
    if is_title_line and header.kind is DocumentKind.SUPPLEMENTARY_AGREEMENT:
        parent = _PARENT_DOCUMENT.match(provision.text)
        if parent is not None:
            yield (
                LinkKind.AMENDS,
                DocumentReference(
                    text=provision.text[parent.start("title") :],
                    title_words=title_words(parent["title"]),
                    number=_number(parent),
                    edition=None,
                ),
                None,
            )
    for repealed in REPEALED_EDITION.finditer(provision.text):
        edition = parse_date(repealed["date"])
        # Та же дата, что у самого документа, — пометка о его собственной отмене:
        # она уже прочитана в шапке.
        if edition is not None and edition != header.edition:
            yield (
                LinkKind.SUPERSEDES,
                DocumentReference(repealed.group(0), (), None, edition),
                None,
            )
    for supplement in _SUPPLEMENT.finditer(provision.text):
        named_from = (
            supplement.start("title")
            if supplement["address"] is None
            else supplement.start("address")
        )
        yield (
            LinkKind.SUPPLEMENTS,
            DocumentReference(
                text=provision.text[named_from : supplement.end()],
                title_words=title_words(supplement["title"]),
                number=_number(supplement),
                edition=(
                    None
                    if supplement["edition"] is None
                    else parse_date(supplement["edition"])
                ),
            ),
            _clause(supplement),
        )


def _number(match: re.Match[str]) -> str | None:
    number: str | None = match["number"]
    return None if number is None else number.rstrip(".")


def _clause(match: re.Match[str]) -> ClauseNumber | None:
    clause: str | None = match["clause"]
    if clause is None:
        return None
    return ClauseNumber(tuple(int(part) for part in clause.split(".")))


def _commencement(
    provisions: Sequence[Provision], header: DocumentHeader
) -> date | None:
    """Дата, с которой действует сказанное в документе.

    None — даты в тексте нет или названо несколько разных: выбирать между ними
    код не берётся.
    """
    if header.effective_from is not None:
        return header.effective_from
    stated: set[date] = set()
    for provision in provisions:
        for line in provision.text.splitlines():
            if header.dated is not None and _FROM_SIGNING.search(line):
                stated.add(header.dated)
            written = [match["date"] for match in _COMMENCEMENT_DATE.finditer(line)]
            leading = _LEADING_DATE.match(line)
            if leading is not None:
                written.append(leading["date"])
            stated.update(
                found for found in map(parse_date, written) if found is not None
            )
    return stated.pop() if len(stated) == 1 else None
