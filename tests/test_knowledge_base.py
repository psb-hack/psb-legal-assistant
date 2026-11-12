import zipfile
from pathlib import Path

import pytest

from lex_documents import (
    DocumentHeader,
    DocumentId,
    DocumentKind,
    LawArticleHeader,
    ProvisionId,
    cite,
)
from lex_errors import KnowledgeBaseError
from lex_knowledge_base import (
    SourceFile,
    company_document,
    load_knowledge_base,
    read_archive,
    read_document,
)

LAW_ARTICLE = (
    "Источник: Статья 300. ТК РФ Учебный отпуск. Трудовой кодекс Российской "
    "Федерации от 30.12.2001 N 197-ФЗ (ТК РФ) (с изменениями и дополнениями)\n"
    "URL: https://example.org/labour/300\n"
    "Дата выгрузки: 05.06.2031\n"
    "Статья 300. Учебный отпуск\n"
    "Учебный отпуск предоставляется работнику по его заявлению.\n"
    "Отпуск оплачивается по среднему заработку.\n"
)

MEMO = (
    "СЛУЖЕБНАЯ ЗАПИСКА № ОК-4 от 20.04.2031\n"
    "О графике отпусков\n"
    "График отпусков утверждается до 15 декабря.\n"
)


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_text_files_are_read_in_order_of_names(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "base.zip",
        {"б.txt": "второй".encode(), "а.md": "первый".encode()},
    )
    assert read_archive(archive) == (
        SourceFile("а.md", "первый"),
        SourceFile("б.txt", "второй"),
    )


def test_folders_service_copies_and_other_formats_are_skipped(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "base.zip",
        {
            "законы/": b"",
            "законы/статья.txt": "норма".encode(),
            "__MACOSX/законы/._статья.txt": b"\x00\x05\x16\x07",
            "законы/._статья.txt": b"\x00\x05\x16\x07",
            "схема.png": b"\x89PNG",
            "договор.docx": b"PK",
        },
    )
    assert [file.name for file in read_archive(archive)] == ["законы/статья.txt"]


def test_decomposed_letters_in_file_name_are_composed(tmp_path: Path) -> None:
    decomposed = "й.txt"
    archive = _archive(tmp_path / "base.zip", {decomposed: b"x"})
    assert read_archive(archive)[0].name == "й.txt"


def test_byte_order_mark_is_not_part_of_the_text(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "base.zip", {"а.txt": "﻿текст".encode()})
    assert read_archive(archive)[0].text == "текст"


def test_file_in_windows_encoding_is_still_read(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "base.zip", {"а.txt": "Договор".encode("cp1251")})
    assert read_archive(archive)[0].text == "Договор"


def test_missing_archive_is_a_knowledge_base_error(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseError, match="не читается"):
        read_archive(tmp_path / "нет.zip")


def test_file_that_is_not_an_archive_is_a_knowledge_base_error(tmp_path: Path) -> None:
    broken = tmp_path / "base.zip"
    broken.write_bytes(b"not a zip")
    with pytest.raises(KnowledgeBaseError, match="не читается"):
        read_archive(broken)


def test_archive_without_text_files_is_a_knowledge_base_error(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "base.zip", {"схема.png": b"\x89PNG"})
    with pytest.raises(KnowledgeBaseError, match="нет текстовых файлов"):
        read_archive(archive)


def test_company_document_joins_header_and_provisions() -> None:
    document = company_document(DocumentId("d07"), SourceFile("записка.txt", MEMO))
    assert document.header.kind is DocumentKind.MEMO
    assert document.header.title == "Служебная записка № ОК-4 от 20.04.2031"
    assert [item.label for item in document.provisions] == [
        "преамбула",
        "преамбула",
        "абз. 1",
    ]
    assert document.provision_id(document.provisions[2]) == "d07.3"


def test_document_without_header_is_named_by_its_file() -> None:
    source = SourceFile("порядок/Порядок выдачи пропусков.txt", "Пропуск выдаётся.\n")
    document = company_document(DocumentId("d01"), source)
    assert document.header.title == "Порядок выдачи пропусков"


def test_law_article_is_recognised_by_its_text_not_by_file_name() -> None:
    document = read_document(
        DocumentId("d02"), SourceFile("прочее/файл.txt", LAW_ARTICLE)
    )
    assert isinstance(document.header, LawArticleHeader)
    assert cite(document, document.provisions[1]) == "ТК РФ, ст. 300, ч. 2"


def test_file_from_laws_folder_without_source_header_is_a_company_document() -> None:
    document = read_document(DocumentId("d02"), SourceFile("законы/записка.txt", MEMO))
    assert isinstance(document.header, DocumentHeader)


def test_documents_are_numbered_in_order_of_file_names(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "base.zip",
        {
            "законы/статья.txt": LAW_ARTICLE.encode(),
            "документы/записка.txt": MEMO.encode(),
        },
    )
    base = load_knowledge_base(archive)
    assert [document.id for document in base.documents] == ["d01", "d02"]
    assert isinstance(base.documents[0].header, DocumentHeader)
    assert isinstance(base.documents[1].header, LawArticleHeader)


def test_provision_is_found_by_its_identifier(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "base.zip",
        {
            "законы/статья.txt": LAW_ARTICLE.encode(),
            "документы/записка.txt": MEMO.encode(),
        },
    )
    base = load_knowledge_base(archive)
    found = base.find(ProvisionId("d02.2"))
    assert found is not None
    document, provision = found
    assert document.id == "d02"
    assert provision.text == "Отпуск оплачивается по среднему заработку."


def test_unknown_identifier_finds_nothing(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "base.zip", {"записка.txt": MEMO.encode()})
    base = load_knowledge_base(archive)
    assert base.find(ProvisionId("d01.99")) is None
    assert base.find(ProvisionId("d07.1")) is None
