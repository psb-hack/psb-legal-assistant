"""Разбор документов компании на положения: нумерованные пункты или абзацы."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lex_document_header import header_lines
from lex_documents import (
    ClauseNumber,
    DocumentPart,
    Heading,
    ParagraphNumber,
    ParsedDocument,
    Provision,
)
from lex_numbering import split_numbered_line
from lex_paragraphs import split_paragraphs

# «7. Реквизиты и подписи сторон» — раздел в конце договора без нумерованных пунктов.
_REQUISITES_TITLE = re.compile(r"реквизиты|подписи", re.IGNORECASE)
# Положение заканчивается знаком препинания, заголовок раздела — нет.
_SENTENCE_ENDINGS = (".", ":", ";")

# Грамматика превращает абзацы документа в положения. Выбирает её строение текста,
# а не жанр: записка с пунктами разбирается как регламент.
_Grammar = Callable[[Sequence[str]], list[Provision]]


@dataclass(frozen=True, slots=True)
class _NumberedBlock:
    """Строка с номером и строки без номера, идущие за ней."""

    number: ClauseNumber
    title: str
    continuation: tuple[str, ...]

    @property
    def reads_as_sentence(self) -> bool:
        return self.title.endswith(_SENTENCE_ENDINGS)

    @property
    def text(self) -> str:
        return "\n".join((self.title, *self.continuation))


def parse(text: str) -> ParsedDocument:
    paragraphs = split_paragraphs(text)
    has_clauses = any(split_numbered_line(line) is not None for line in paragraphs)
    grammar: _Grammar = _numbered_clauses if has_clauses else _plain_paragraphs
    return ParsedDocument(provisions=tuple(grammar(paragraphs)))


def _numbered_clauses(paragraphs: Sequence[str]) -> list[Provision]:
    preamble, blocks = _split_into_blocks(paragraphs)
    provisions = [
        Provision(
            position=position,
            part=DocumentPart.PREAMBLE,
            number=None,
            headings=(),
            text=line,
        )
        for position, line in enumerate(preamble, start=1)
    ]
    ancestors: list[_NumberedBlock] = []
    for index, block in enumerate(blocks):
        while ancestors and not ancestors[-1].number.is_ancestor_of(block.number):
            ancestors.pop()
        has_subclauses = index + 1 < len(blocks) and block.number.is_ancestor_of(
            blocks[index + 1].number
        )
        # Пункт с подпунктами — заголовок для них. Если в нём самом есть текст
        # нормы («Покупатель обязан:»), он остаётся и положением, иначе норма теряется.
        if not has_subclauses or block.reads_as_sentence or block.continuation:
            provisions.append(
                Provision(
                    position=len(provisions) + 1,
                    part=_part_of(block, has_subclauses),
                    number=block.number,
                    headings=tuple(
                        Heading(ancestor.number, ancestor.title)
                        for ancestor in ancestors
                    ),
                    text=block.text,
                )
            )
        if has_subclauses:
            ancestors.append(block)
    return provisions


def _plain_paragraphs(paragraphs: Sequence[str]) -> list[Provision]:
    """Шапка остаётся преамбулой, абзацы основного текста нумеруются с первого."""
    header_length = len(header_lines(paragraphs))
    provisions: list[Provision] = []
    for position, line in enumerate(paragraphs, start=1):
        in_header = position <= header_length
        provisions.append(
            Provision(
                position=position,
                part=DocumentPart.PREAMBLE if in_header else DocumentPart.BODY,
                number=None if in_header else ParagraphNumber(position - header_length),
                headings=(),
                text=line,
            )
        )
    return provisions


def _split_into_blocks(
    paragraphs: Sequence[str],
) -> tuple[list[str], list[_NumberedBlock]]:
    preamble: list[str] = []
    blocks: list[_NumberedBlock] = []
    for line in paragraphs:
        numbered = split_numbered_line(line)
        if numbered is not None:
            blocks.append(_NumberedBlock(numbered.number, numbered.text, ()))
        elif blocks:
            last = blocks[-1]
            blocks[-1] = _NumberedBlock(
                last.number, last.title, (*last.continuation, line)
            )
        else:
            preamble.append(line)
    return preamble, blocks


def _part_of(block: _NumberedBlock, has_subclauses: bool) -> DocumentPart:
    is_bare_heading = not has_subclauses and not block.reads_as_sentence
    if is_bare_heading and _REQUISITES_TITLE.search(block.title):
        return DocumentPart.REQUISITES
    return DocumentPart.BODY
