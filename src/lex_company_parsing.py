"""Разбор документов компании с нумерованными положениями: договоры, регламенты."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lex_documents import (
    ClauseNumber,
    DocumentPart,
    Heading,
    ParsedDocument,
    Provision,
)
from lex_numbering import split_numbered_line
from lex_paragraphs import split_paragraphs

# «7. Реквизиты и подписи сторон» — раздел в конце договора без нумерованных пунктов.
_REQUISITES_TITLE = re.compile(r"реквизиты|подписи", re.IGNORECASE)
# Положение заканчивается знаком препинания, заголовок раздела — нет.
_SENTENCE_ENDINGS = (".", ":", ";")


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
    preamble, blocks = _split_into_blocks(text)
    provisions = [
        Provision(part=DocumentPart.PREAMBLE, number=None, headings=(), text=line)
        for line in preamble
    ]
    ancestors: list[Heading] = []
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
                    part=_part_of(block, has_subclauses),
                    number=block.number,
                    headings=tuple(ancestors),
                    text=block.text,
                )
            )
        if has_subclauses:
            ancestors.append(Heading(number=block.number, title=block.title))
    return ParsedDocument(provisions=tuple(provisions))


def _split_into_blocks(text: str) -> tuple[list[str], list[_NumberedBlock]]:
    preamble: list[str] = []
    blocks: list[_NumberedBlock] = []
    for line in split_paragraphs(text):
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
