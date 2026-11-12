"""Чтение DOCX: видимый текст отдельно от того, что читатель документа не видит."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import docx
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from lex_documents import HiddenContent, HiddenContentKind
from lex_errors import KnowledgeBaseError

_CELL_SEPARATOR = " | "


@dataclass(frozen=True, slots=True)
class DocxContent:
    # Абзацы и строки таблиц в порядке документа, по одному на строку.
    text: str
    hidden: tuple[HiddenContent, ...]


def read_docx(name: str, raw: bytes) -> DocxContent:
    try:
        document = docx.Document(io.BytesIO(raw))
    except (zipfile.BadZipFile, PackageNotFoundError, KeyError, ValueError) as error:
        raise KnowledgeBaseError(f"файл DOCX не читается: {name}") from error
    visible, vanished = _body(document)
    hidden = [
        *(HiddenContent(HiddenContentKind.HIDDEN_TEXT, text) for text in vanished),
        *(
            HiddenContent(HiddenContentKind.COMMENT, comment.text)
            for comment in document.comments
        ),
        *(
            HiddenContent(HiddenContentKind.HEADER_FOOTER, text)
            for text in _headers_and_footers(document)
        ),
        *(
            HiddenContent(HiddenContentKind.PROPERTY, text)
            for text in _properties(document)
        ),
    ]
    return DocxContent(
        text="\n".join(visible),
        hidden=tuple(item for item in hidden if item.text.strip()),
    )


def _body(document: DocxDocument) -> tuple[list[str], list[str]]:
    visible: list[str] = []
    vanished: list[str] = []
    for block in document.iter_inner_content():
        rows = [[[block]]] if isinstance(block, Paragraph) else _rows_of(block)
        for cells in rows:
            shown_cells: list[str] = []
            for paragraphs in cells:
                parts = [_split_by_visibility(item) for item in paragraphs]
                shown_cells.append(" ".join(shown for shown, _ in parts if shown))
                vanished.extend(concealed for _, concealed in parts if concealed)
            visible.append(_CELL_SEPARATOR.join(shown_cells))
    return visible, vanished


def _rows_of(table: Table) -> list[list[list[Paragraph]]]:
    """Строки таблицы: в каждой — абзацы ячеек; объединённая ячейка читается раз."""
    rows: list[list[list[Paragraph]]] = []
    for row in table.rows:
        seen: set[int] = set()
        cells: list[list[Paragraph]] = []
        for cell in row.cells:
            if id(cell._tc) not in seen:
                seen.add(id(cell._tc))
                cells.append(list(cell.paragraphs))
        rows.append(cells)
    return rows


def _split_by_visibility(paragraph: Paragraph) -> tuple[str, str]:
    shown: list[str] = []
    concealed: list[str] = []
    for item in paragraph.iter_inner_content():
        for run in [item] if isinstance(item, Run) else item.runs:
            (concealed if _is_hidden(run, paragraph) else shown).append(run.text)
    return "".join(shown), "".join(concealed)


def _is_hidden(run: Run, paragraph: Paragraph) -> bool:
    """Атрибут w:vanish у фрагмента, у его стиля знаков или у стиля абзаца."""
    fonts = [run.font]
    if run.style is not None:
        fonts.append(run.style.font)
    if paragraph.style is not None:
        fonts.append(paragraph.style.font)
    for font in fonts:
        if font.hidden is not None:
            return font.hidden
    return False


def _headers_and_footers(document: DocxDocument) -> list[str]:
    texts: list[str] = []
    for section in document.sections:
        for part in (
            section.header,
            section.first_page_header,
            section.even_page_header,
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ):
            # Колонтитул, унаследованный от предыдущего раздела, уже прочитан там.
            if not part.is_linked_to_previous:
                texts.append("\n".join(item.text for item in part.paragraphs))
    return texts


def _properties(document: DocxDocument) -> list[str]:
    core = document.core_properties
    return [
        core.title,
        core.subject,
        core.keywords,
        core.comments,
        core.category,
        core.content_status,
        core.author,
        core.last_modified_by,
    ]
