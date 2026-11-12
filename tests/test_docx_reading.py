import io
import zipfile
from pathlib import Path

import docx
import pytest
from docx.document import Document as DocxDocument

from lex_documents import DocumentId, HiddenContent, HiddenContentKind
from lex_docx_reading import read_docx
from lex_errors import KnowledgeBaseError
from lex_knowledge_base import load_knowledge_base, read_archive

HIDDEN_NOTE = "Этот абзац читателю не показан."


def _saved(document: DocxDocument) -> bytes:
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _memo() -> DocxDocument:
    document = docx.Document()
    document.add_paragraph("СЛУЖЕБНАЯ ЗАПИСКА № ОК-4 от 20.04.2031")
    document.add_paragraph("О графике отпусков")
    document.add_paragraph("График отпусков утверждается до 15 декабря.")
    return document


def _hidden(raw: bytes, kind: HiddenContentKind) -> list[str]:
    return [item.text for item in read_docx("а.docx", raw).hidden if item.kind is kind]


def test_paragraphs_become_lines_of_text() -> None:
    assert read_docx("а.docx", _saved(_memo())).text == (
        "СЛУЖЕБНАЯ ЗАПИСКА № ОК-4 от 20.04.2031\n"
        "О графике отпусков\n"
        "График отпусков утверждается до 15 декабря."
    )


def test_table_rows_are_read_in_place_between_paragraphs() -> None:
    document = docx.Document()
    document.add_paragraph("Перечень отпусков:")
    table = document.add_table(rows=2, cols=2)
    for row, cells in zip(
        table.rows, (("Отдел", "Месяц"), ("Склад", "июль")), strict=True
    ):
        for cell, text in zip(row.cells, cells, strict=True):
            cell.text = text
    document.add_paragraph("Перечень действует один год.")
    assert read_docx("а.docx", _saved(document)).text.splitlines() == [
        "Перечень отпусков:",
        "Отдел | Месяц",
        "Склад | июль",
        "Перечень действует один год.",
    ]


def test_merged_cell_is_read_once() -> None:
    document = docx.Document()
    table = document.add_table(rows=1, cols=3)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Склад"
    table.cell(0, 2).text = "июль"
    assert read_docx("а.docx", _saved(document)).text == "Склад | июль"


def test_hidden_run_is_kept_apart_from_visible_text() -> None:
    document = _memo()
    paragraph = document.add_paragraph("Отпуск делится на части.")
    paragraph.add_run(HIDDEN_NOTE).font.hidden = True
    content = read_docx("а.docx", _saved(document))
    assert content.text.splitlines()[-1] == "Отпуск делится на части."
    assert HiddenContent(HiddenContentKind.HIDDEN_TEXT, HIDDEN_NOTE) in content.hidden


def test_run_hidden_by_its_style_is_hidden_text() -> None:
    document = _memo()
    document.styles["Emphasis"].font.hidden = True
    document.add_paragraph("Видимая часть.").add_run(HIDDEN_NOTE, style="Emphasis")
    raw = _saved(document)
    assert read_docx("а.docx", raw).text.splitlines()[-1] == "Видимая часть."
    assert _hidden(raw, HiddenContentKind.HIDDEN_TEXT) == [HIDDEN_NOTE]


def test_comments_headers_and_properties_are_hidden_content() -> None:
    document = _memo()
    target = document.paragraphs[2].runs
    document.add_comment(target, text="Комментарий рецензента.", author="К.")
    section = document.sections[0]
    section.header.paragraphs[0].text = "Верхний колонтитул"
    section.footer.paragraphs[0].text = "Нижний колонтитул"
    document.core_properties.subject = "Тема из свойств"
    raw = _saved(document)
    assert _hidden(raw, HiddenContentKind.COMMENT) == ["Комментарий рецензента."]
    assert _hidden(raw, HiddenContentKind.HEADER_FOOTER) == [
        "Верхний колонтитул",
        "Нижний колонтитул",
    ]
    assert "Тема из свойств" in _hidden(raw, HiddenContentKind.PROPERTY)
    visible = read_docx("а.docx", raw).text
    for concealed in ("Комментарий", "колонтитул", "Тема из свойств"):
        assert concealed not in visible


def test_file_that_is_not_a_docx_is_a_knowledge_base_error() -> None:
    with pytest.raises(KnowledgeBaseError, match=r"не читается: договор\.docx"):
        read_docx("договор.docx", b"PK")


def test_docx_in_archive_is_parsed_like_a_text_document(tmp_path: Path) -> None:
    document = _memo()
    document.paragraphs[2].add_run(HIDDEN_NOTE).font.hidden = True
    archive = tmp_path / "base.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        packed.writestr("записка.docx", _saved(document))
    assert read_archive(archive)[0].name == "записка.docx"
    loaded = load_knowledge_base(archive).documents[0]
    assert loaded.id == DocumentId("d01")
    assert loaded.header.title == "Служебная записка № ОК-4 от 20.04.2031"
    assert [item.text for item in loaded.provisions][-1] == (
        "График отпусков утверждается до 15 декабря."
    )
    assert HiddenContent(HiddenContentKind.HIDDEN_TEXT, HIDDEN_NOTE) in loaded.hidden
