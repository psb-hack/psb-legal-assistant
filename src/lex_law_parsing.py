"""Разбор статьи закона на положения с адресами, принятыми в самом акте."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from lex_documents import (
    AmendmentNote,
    DocumentPart,
    Heading,
    LawAddress,
    LawArticleHeader,
    LawUnit,
    LawUnitKind,
    ParagraphRange,
    Provision,
    RepealedUnit,
)
from lex_errors import KnowledgeBaseError
from lex_law_lines import (
    ArticleTitle,
    LawLine,
    LawLineKind,
    MarkerClosing,
    UnitMarker,
    classify_lines,
    is_repealed_part,
    read_article_title,
    repealed_paragraph_count,
    split_marker,
)
from lex_law_notes import read_amendment, read_source
from lex_paragraphs import split_paragraphs

# Единица длиннее делится по абзацам на несколько положений с тем же адресом.
# В 13 статьях базы короче 218 единиц из 243; все 25 более длинных состоят
# из нескольких абзацев — это перечни и многоабзацные пункты НК.
MAX_PROVISION_LENGTH = 2000

# Единицу «1.» называют пунктом в этих кодексах и частью — в остальных актах.
_CODES_WITH_CLAUSES = ("налоговый кодекс", "гражданский кодекс")
# «Примечания:» — после этой строки в статье КоАП нумерация начинается заново.
_NOTES_TITLE = re.compile(r"^Примечания:$")
# Делить можно после законченного абзаца или элемента перечня. После «:» и «-»
# предложение продолжается: вводная фраза и первый элемент, диспозиция и санкция.
_BREAKABLE_ENDINGS = (".", ";")
_INTRODUCTION_ENDING = ":"


@dataclass(frozen=True, slots=True)
class ParsedArticle:
    header: LawArticleHeader
    provisions: tuple[Provision, ...]


@dataclass(frozen=True, slots=True)
class _Addressing:
    """Как акт называет свои единицы и где в статье начинается новая."""

    dotted: LawUnitKind
    # В ТК и части федеральных законов части не нумеруются: новая часть —
    # абзац с прописной буквы, абзацы перечня начинаются со строчной.
    unnumbered_parts: bool


@dataclass(slots=True)
class _Unit:
    address: tuple[LawUnit, ...]
    parent: _Unit | None
    # Номер абзаца в единице и его текст. Абзацы, утратившие силу, текста
    # не дают, но номер занимают.
    paragraphs: list[tuple[int, str]] = field(default_factory=list)
    next_paragraph: int = 1

    def add(self, text: str) -> None:
        self.paragraphs.append((self.next_paragraph, text))
        self.next_paragraph += 1

    def skip(self, count: int) -> ParagraphRange:
        first = self.next_paragraph
        self.next_paragraph += count
        return ParagraphRange(first, self.next_paragraph - 1)


class _ArticleBody:
    """Собирает единицы статьи, пометки и заглушки по мере чтения строк."""

    def __init__(self, addressing: _Addressing) -> None:
        self._addressing = addressing
        self.units: list[_Unit] = []
        self.amendments: list[AmendmentNote] = []
        self.comments: list[str] = []
        self.repealed: list[RepealedUnit] = []
        # Пометка стоит перед своей единицей: адрес узнаём, когда единица открыта.
        self._awaiting_unit: list[int] = []
        self._dotted: _Unit | None = None
        self._current: _Unit | None = None
        self._notes: _Unit | None = None
        self._parts = 0

    def add_comment(self, text: str) -> None:
        note = read_amendment(text)
        if note is None:
            self.comments.append(text)
            return
        if note.target is not None:
            self._awaiting_unit.append(len(self.amendments))
        self.amendments.append(note)

    def add_norm(self, text: str) -> None:
        if _NOTES_TITLE.match(text):
            self._notes = self._open((LawUnit(LawUnitKind.NOTE, None),), None)
            self._dotted = None
            self._notes.add(text)
            return
        marker = split_marker(text)
        if marker is not None:
            self._open_numbered(marker).add(marker.text)
        elif self._starts_part(text):
            self._open_part().add(text)
        elif self._current is None:
            # Текст до первой единицы статьи адреса не имеет.
            self._open((), None).add(text)
        else:
            self._current.add(text)

    def add_repealed(self, text: str) -> None:
        marker = split_marker(text)
        paragraphs = repealed_paragraph_count(text)
        if marker is not None:
            address = LawAddress(self._open_numbered(marker).address, None)
        elif self._addressing.unnumbered_parts and is_repealed_part(text):
            address = LawAddress(self._open_part().address, None)
        elif paragraphs is not None and self._current is not None:
            address = LawAddress(self._current.address, self._current.skip(paragraphs))
        else:
            # Заглушка без места в строении статьи остаётся комментарием.
            self.comments.append(text)
            return
        self.repealed.append(RepealedUnit(address, text))

    def _starts_part(self, text: str) -> bool:
        if not self._addressing.unnumbered_parts or self._notes is not None:
            return False
        first_letter = next((char for char in text if char.isalpha()), "")
        return first_letter.isupper()

    def _open_part(self) -> _Unit:
        self._parts += 1
        part = self._open((LawUnit(LawUnitKind.PART, str(self._parts)),), None)
        self._dotted = part
        return part

    def _open_numbered(self, marker: UnitMarker) -> _Unit:
        if marker.closing is MarkerClosing.BRACKET:
            parent = self._dotted
            inside_clause = (
                parent is not None and parent.address[-1].kind is LawUnitKind.CLAUSE
            )
            kind = LawUnitKind.SUBCLAUSE if inside_clause else LawUnitKind.CLAUSE
            outer = () if parent is None else parent.address
            return self._open((*outer, LawUnit(kind, marker.number)), parent)
        kind = self._addressing.dotted if self._notes is None else LawUnitKind.NOTE
        self._dotted = self._open((LawUnit(kind, marker.number),), self._notes)
        return self._dotted

    def _open(self, address: tuple[LawUnit, ...], parent: _Unit | None) -> _Unit:
        unit = _Unit(address, parent)
        self.units.append(unit)
        self._current = unit
        for index in self._awaiting_unit:
            note = self.amendments[index]
            named = None if note.target is None else note.target.units[-1]
            if address and named == address[-1]:
                self.amendments[index] = replace(note, target=LawAddress(address, None))
        self._awaiting_unit.clear()
        return unit


def parse_article(text: str) -> ParsedArticle:
    lines = classify_lines(split_paragraphs(text))
    title_index, title = _find_title(lines)
    source = read_source(line.text for line in lines if line.kind is LawLineKind.SOURCE)
    following = lines[title_index + 1 :]
    dotted = (
        LawUnitKind.CLAUSE
        if source.act.lower().startswith(_CODES_WITH_CLAUSES)
        else LawUnitKind.PART
    )
    body = _ArticleBody(_Addressing(dotted, not _has_numbered_units(following)))
    for line in lines[:title_index]:
        # До заголовка статьи норм нет: всё, кроме шапки, — комментарии.
        if line.kind is not LawLineKind.SOURCE:
            body.add_comment(line.text)
    for line in following:
        if line.kind is LawLineKind.NORM:
            body.add_norm(line.text)
        elif line.kind is LawLineKind.REPEALED:
            body.add_repealed(line.text)
        else:
            body.add_comment(line.text)
    header = LawArticleHeader(
        act=source.act,
        act_short=source.act_short,
        article=title.number,
        title=title.title,
        source_url=source.url,
        retrieved=source.retrieved,
        amendments=tuple(body.amendments),
        comments=tuple(body.comments),
        repealed=tuple(body.repealed),
    )
    return ParsedArticle(header, tuple(_provisions(body.units)))


def _find_title(lines: Sequence[LawLine]) -> tuple[int, ArticleTitle]:
    for index, line in enumerate(lines):
        if line.kind is LawLineKind.ARTICLE_TITLE:
            title = read_article_title(line.text)
            if title is not None:
                return index, title
    raise KnowledgeBaseError("в тексте нет строки «Статья N.»")


def _has_numbered_units(lines: Sequence[LawLine]) -> bool:
    """Есть ли единицы «1.» до раздела примечаний: у примечаний нумерация своя."""
    for line in lines:
        if _NOTES_TITLE.match(line.text):
            return False
        marker = split_marker(line.text)
        is_unit = line.kind in (LawLineKind.NORM, LawLineKind.REPEALED)
        if is_unit and marker is not None and marker.closing is MarkerClosing.DOT:
            return True
    return False


def _provisions(units: Sequence[_Unit]) -> list[Provision]:
    provisions: list[Provision] = []
    for unit in units:
        if not unit.paragraphs:
            continue
        chunks = _chunks(unit.paragraphs)
        for index, chunk in enumerate(chunks):
            span = (
                None if len(chunks) == 1 else ParagraphRange(chunk[0][0], chunk[-1][0])
            )
            provisions.append(
                Provision(
                    position=len(provisions) + 1,
                    part=DocumentPart.BODY if unit.address else DocumentPart.PREAMBLE,
                    number=LawAddress(unit.address, span) if unit.address else None,
                    headings=_headings(unit, continues_unit=index > 0),
                    text="\n".join(text for _, text in chunk),
                )
            )
    return provisions


def _chunks(paragraphs: Sequence[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    chunks = [[paragraphs[0]]]
    for paragraph in paragraphs[1:]:
        current = chunks[-1]
        length = sum(len(text) for _, text in current) + len(paragraph[1])
        can_break = current[-1][1].endswith(_BREAKABLE_ENDINGS)
        if length > MAX_PROVISION_LENGTH and can_break:
            chunks.append([paragraph])
        else:
            current.append(paragraph)
    return chunks


def _headings(unit: _Unit, continues_unit: bool) -> tuple[Heading, ...]:
    """Вводные фразы родителей; у продолжения поделённой единицы — и её собственная."""
    path: list[Heading] = []
    introduction = unit.paragraphs[0][1]
    if continues_unit and unit.address and introduction.endswith(_INTRODUCTION_ENDING):
        path.append(Heading(unit.address[-1], introduction))
    ancestor = unit.parent
    while ancestor is not None:
        if ancestor.paragraphs:
            path.append(Heading(ancestor.address[-1], ancestor.paragraphs[0][1]))
        ancestor = ancestor.parent
    return tuple(reversed(path))
