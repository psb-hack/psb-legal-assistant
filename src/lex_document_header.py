"""Метаданные документа компании из его шапки: вид, номер, даты, стороны, статус."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from lex_dates import DATE_PATTERN, parse_date
from lex_documents import DocumentHeader, DocumentKind, DocumentStatus, Party
from lex_numbering import split_numbered_line
from lex_paragraphs import split_paragraphs

_KIND_BY_GENRE = {
    "дополнительное соглашение": DocumentKind.SUPPLEMENTARY_AGREEMENT,
    "договор": DocumentKind.CONTRACT,
    "контракт": DocumentKind.CONTRACT,
    "регламент": DocumentKind.REGULATION,
    "положение": DocumentKind.REGULATION,
    "служебная записка": DocumentKind.MEMO,
    "докладная записка": DocumentKind.MEMO,
    "письмо": DocumentKind.LETTER,
}
_LETTER_TITLE = "Письмо"
_LEGAL_FORMS = ("ООО", "АО", "ПАО", "ЗАО", "ОАО", "НАО", "ИП")
_NUMBER_SIGN = "№"
_SENTENCE_ENDINGS = (".", "!", "?", ":", ";")

# «Исх. № К-12 от 15.04.2025» — слова «письмо» в самом письме обычно нет.
_OUTGOING_NUMBER = re.compile(rf"^исх\.?\s*(?={_NUMBER_SIGN})", re.IGNORECASE)
# «к договору поставки № П-17/2024 от 12 февраля 2024 г.» — вторая строка названия.
_TITLE_CONTINUATION = re.compile(r"^к\s", re.IGNORECASE)
# «Договор поставки № П-17/2024» → «П-17/2024»
_NUMBER = re.compile(rf"{_NUMBER_SIGN}\s*([^\s,;]+)")
# «Служебная записка № ЮО-7 от 12.05.2025»
_NUMBER_AND_DATE = re.compile(
    rf"{_NUMBER_SIGN}\s*\S+\s+от\s+(?P<date>{DATE_PATTERN})", re.IGNORECASE
)
# «г. Москва — 12 февраля 2024 г.»
_PLACE_AND_DATE = re.compile(
    rf"^.+\s[—–-]\s+(?P<date>{DATE_PATTERN})(?:\s*г\.?)?$", re.IGNORECASE
)
# «Утверждён приказом генерального директора № 2-П от 10.01.2023»
_APPROVAL = re.compile(
    rf"^утвержд[её]н[аоы]?\s.*?\sот\s+(?P<date>{DATE_PATTERN})", re.IGNORECASE
)
# «… от 12 февраля 2024 г.» — точка после «г» не конец предложения.
_ENDS_WITH_DATE = re.compile(rf"{DATE_PATTERN}\s*г\.$", re.IGNORECASE)
# «Редакция от 01.03.2025.»
_EDITION = re.compile(rf"^редакция\s+от\s+(?P<date>{DATE_PATTERN})\.?$", re.IGNORECASE)
# «Действует с 01.03.2025.»
_EFFECTIVE_FROM = re.compile(
    r"(?:действует|вступает\s+в\s+силу|вводится\s+в\s+действие)"
    rf"\s+с\s+(?P<date>{DATE_PATTERN})",
    re.IGNORECASE,
)
_REPEAL_VERB = r"утра(?:тил\w*|чивает)\s+силу"
# «Редакция утратила силу с 01.03.2025 в связи с …»
_REPEAL = re.compile(
    rf"{_REPEAL_VERB}(?:\s+с\s+(?P<date>{DATE_PATTERN}))?", re.IGNORECASE
)
# «Редакция от 10.01.2023 утрачивает силу.» — силу теряет названная редакция.
_REPEALED_EDITION = re.compile(
    rf"редакци\w+\s+от\s+(?P<date>{DATE_PATTERN})(?:\s*г\.)?\s+{_REPEAL_VERB}",
    re.IGNORECASE,
)
# «Редакция от 01.03.2025. Действует с 01.03.2025.» → два утверждения.
_SENTENCE_BREAK = re.compile(r"(?<=\.)\s+(?=[А-ЯЁ])")
# «ООО «Северный Склад» (ИНН 5047112233), именуемое в дальнейшем «Арендодатель», …»
_FULL_PARTY = re.compile(
    r"(?P<name>[А-ЯЁA-Z][^,;()]*?)\s*(?:\(ИНН\s+(?P<inn>\d{12}|\d{10})\))?"
    r",\s+именуем\w+\s+в\s+дальнейшем\s+«(?P<role>[^»]+)»"
)
# «ООО «Альфа-Трейд» («Поставщик») в лице …»
_SHORT_PARTY = re.compile(
    rf"(?P<name>(?:{'|'.join(_LEGAL_FORMS)})\s+«[^»]+»)\s+\(«(?P<role>[^»]+)»\)"
)
_HEADER_STATEMENTS = (
    _ENDS_WITH_DATE,
    _EDITION,
    _EFFECTIVE_FROM,
    _REPEAL,
    _FULL_PARTY,
    _SHORT_PARTY,
)


@dataclass(frozen=True, slots=True)
class _TitleLines:
    kind: DocumentKind
    own: str
    continuation: str | None


def read_header(text: str) -> DocumentHeader:
    lines = _header_lines(text)
    title = _find_title(lines)
    statements = [
        statement for line in lines for statement in _SENTENCE_BREAK.split(line)
    ]
    edition = _first_date(_EDITION, statements)
    dated = _dated(lines, title.own)
    repeal = _own_repeal(statements, edition, dated)
    return DocumentHeader(
        kind=title.kind,
        title=_readable_title(title),
        number=_number(title.own),
        dated=dated,
        parties=_parties(lines),
        edition=edition,
        effective_from=_first_date(_EFFECTIVE_FROM, statements),
        status=DocumentStatus.IN_FORCE if repeal is None else DocumentStatus.REPEALED,
        repealed_from=None if repeal is None else _first_date(_REPEAL, [repeal]),
    )


def _header_lines(text: str) -> list[str]:
    """Строки до первого нумерованного пункта или первого абзаца основного текста."""
    lines: list[str] = []
    for line in split_paragraphs(text):
        if split_numbered_line(line) is not None or _is_body_text(line):
            break
        lines.append(line)
    return lines


def _is_body_text(line: str) -> bool:
    if not line.endswith(_SENTENCE_ENDINGS):
        return False
    return not any(pattern.search(line) for pattern in _HEADER_STATEMENTS)


def _find_title(lines: list[str]) -> _TitleLines:
    for index, line in enumerate(lines):
        kind = _kind_of(line)
        if kind is not DocumentKind.UNKNOWN:
            following = lines[index + 1] if index + 1 < len(lines) else ""
            continues = _TITLE_CONTINUATION.match(following) is not None
            return _TitleLines(kind, line, following if continues else None)
    return _TitleLines(DocumentKind.UNKNOWN, lines[0] if lines else "", None)


def _kind_of(line: str) -> DocumentKind:
    if _OUTGOING_NUMBER.match(line):
        return DocumentKind.LETTER
    lowered = line.lower()
    for genre, kind in _KIND_BY_GENRE.items():
        if lowered == genre or lowered.startswith(f"{genre} "):
            return kind
    return DocumentKind.UNKNOWN


def _readable_title(title: _TitleLines) -> str:
    outgoing = _OUTGOING_NUMBER.match(title.own)
    if outgoing is not None:
        return f"{_LETTER_TITLE} {title.own[outgoing.end() :]}"
    own = _without_capitals(title.own)
    readable = own[:1].upper() + own[1:]
    if title.continuation is None:
        return readable
    return f"{readable} {_without_capitals(title.continuation)}"


def _without_capitals(line: str) -> str:
    """«ДОГОВОР ПОСТАВКИ № П-17/2024» → «договор поставки № П-17/2024»."""
    words = line.split(" ")
    if not words[0].isupper():
        return line
    cased: list[str] = []
    for index, word in enumerate(words):
        follows_number_sign = index > 0 and words[index - 1] == _NUMBER_SIGN
        if follows_number_sign or word in _LEGAL_FORMS or not word.isupper():
            cased.append(word)
        elif word.startswith("«"):
            cased.append("«" + word[1:].capitalize())
        else:
            cased.append(word.lower())
    return " ".join(cased)


def _number(title_line: str) -> str | None:
    match = _NUMBER.search(title_line)
    return None if match is None else match.group(1).rstrip(".")


def _dated(lines: list[str], title_line: str) -> date | None:
    return (
        _first_date(_PLACE_AND_DATE, lines)
        or _first_date(_NUMBER_AND_DATE, [title_line])
        or _first_date(_APPROVAL, lines)
    )


def _first_date(pattern: re.Pattern[str], fragments: Iterable[str]) -> date | None:
    for fragment in fragments:
        match = pattern.search(fragment)
        if match is not None and match.group("date") is not None:
            found = parse_date(match.group("date"))
            if found is not None:
                return found
    return None


def _own_repeal(
    statements: list[str], edition: date | None, dated: date | None
) -> str | None:
    """Утверждение об утрате силы самим документом, а не другой редакцией."""
    for statement in statements:
        if _REPEAL.search(statement) is None:
            continue
        repealed_edition = _first_date(_REPEALED_EDITION, [statement])
        if repealed_edition is None or repealed_edition in (edition, dated):
            return statement
    return None


def _parties(lines: list[str]) -> tuple[Party, ...]:
    parties: list[Party] = []
    for line in lines:
        parties.extend(
            Party(match["name"], match["inn"], match["role"])
            for match in _FULL_PARTY.finditer(line)
        )
        parties.extend(
            Party(match["name"], None, match["role"])
            for match in _SHORT_PARTY.finditer(line)
        )
    return tuple(parties)
