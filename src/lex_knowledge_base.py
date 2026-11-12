"""Чтение архива базы знаний и сборка документов из текстов его файлов."""

from __future__ import annotations

import unicodedata
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from lex_company_parsing import parse
from lex_document_header import read_header
from lex_documents import (
    Document,
    DocumentId,
    HiddenContent,
    Provision,
    ProvisionId,
)
from lex_docx_reading import read_docx
from lex_errors import KnowledgeBaseError
from lex_law_lines import is_law_article
from lex_law_parsing import parse_article

_TEXT_SUFFIXES = (".txt", ".md")
_DOCX_SUFFIX = ".docx"
# Архиватор macOS кладёт рядом с файлами служебные копии: «__MACOSX/._договор.txt».
_MACOS_SERVICE_DIRECTORY = "__MACOSX/"
_MACOS_SERVICE_COPY_PREFIX = "._"
_FALLBACK_ENCODING = "cp1251"


@dataclass(frozen=True, slots=True)
class SourceFile:
    name: str
    text: str
    # Скрытый текст, комментарии, колонтитулы и свойства DOCX.
    hidden: tuple[HiddenContent, ...] = ()


def read_archive(path: Path) -> tuple[SourceFile, ...]:
    """Документы архива (TXT, MD, DOCX) в порядке имён."""
    try:
        with zipfile.ZipFile(path) as archive:
            files = [
                _source_file(info.filename, archive.read(info))
                for info in archive.infolist()
                if _is_document_file(info)
            ]
    except (OSError, zipfile.BadZipFile) as error:
        raise KnowledgeBaseError(f"архив базы не читается: {path.name}") from error
    if not files:
        raise KnowledgeBaseError(f"в архиве базы нет документов: {path.name}")
    return tuple(sorted(files, key=lambda file: file.name))


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    documents: tuple[Document, ...]

    def find(self, provision_id: ProvisionId) -> tuple[Document, Provision] | None:
        """Положение и его документ; None — такого идентификатора в базе нет."""
        for document in self.documents:
            for provision in document.provisions:
                if document.provision_id(provision) == provision_id:
                    return document, provision
        return None


def load_knowledge_base(path: Path) -> KnowledgeBase:
    """Документы архива с идентификаторами по порядку имён файлов: «d01», «d02»…"""
    return KnowledgeBase(
        tuple(
            read_document(DocumentId(f"d{index:02d}"), source)
            for index, source in enumerate(read_archive(path), start=1)
        )
    )


def read_document(document_id: DocumentId, source: SourceFile) -> Document:
    # Вид документа определяет текст, а не каталог или имя файла в архиве.
    if is_law_article(source.text):
        return law_article(document_id, source)
    return company_document(document_id, source)


def law_article(document_id: DocumentId, source: SourceFile) -> Document:
    article = parse_article(source.text)
    return Document(
        id=document_id,
        header=article.header,
        provisions=article.provisions,
        hidden=source.hidden,
    )


def company_document(document_id: DocumentId, source: SourceFile) -> Document:
    header = read_header(source.text)
    if not header.title:
        # В тексте нет шапки: называем документ по файлу, иначе на него не сослаться.
        header = replace(header, title=PurePosixPath(source.name).stem)
    return Document(
        id=document_id,
        header=header,
        provisions=parse(source.text).provisions,
        hidden=source.hidden,
    )


def _source_file(archived_name: str, raw: bytes) -> SourceFile:
    name = _normalized(archived_name)
    if PurePosixPath(name).suffix.lower() == _DOCX_SUFFIX:
        content = read_docx(name, raw)
        return SourceFile(name, content.text, content.hidden)
    return SourceFile(name, _decoded(raw))


def _is_document_file(info: zipfile.ZipInfo) -> bool:
    name = PurePosixPath(info.filename)
    if info.is_dir() or name.suffix.lower() not in (*_TEXT_SUFFIXES, _DOCX_SUFFIX):
        return False
    is_service_copy = info.filename.startswith(
        _MACOS_SERVICE_DIRECTORY
    ) or name.name.startswith(_MACOS_SERVICE_COPY_PREFIX)
    return not is_service_copy


def _normalized(name: str) -> str:
    # macOS хранит «й» и «ё» как букву и отдельный знак; приводим к одному виду.
    return unicodedata.normalize("NFC", name)


def _decoded(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode(_FALLBACK_ENCODING, errors="replace")
