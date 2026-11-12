"""Чтение архива базы знаний и сборка документа компании из его текста."""

from __future__ import annotations

import unicodedata
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from lex_company_parsing import parse
from lex_document_header import read_header
from lex_documents import Document, DocumentId
from lex_errors import KnowledgeBaseError

_TEXT_SUFFIXES = (".txt", ".md")
# Архиватор macOS кладёт рядом с файлами служебные копии: «__MACOSX/._договор.txt».
_MACOS_SERVICE_DIRECTORY = "__MACOSX/"
_MACOS_SERVICE_COPY_PREFIX = "._"
_FALLBACK_ENCODING = "cp1251"


@dataclass(frozen=True, slots=True)
class SourceFile:
    name: str
    text: str


def read_archive(path: Path) -> tuple[SourceFile, ...]:
    """Текстовые файлы архива в порядке имён."""
    try:
        with zipfile.ZipFile(path) as archive:
            files = [
                SourceFile(_normalized(info.filename), _decoded(archive.read(info)))
                for info in archive.infolist()
                if _is_text_file(info)
            ]
    except (OSError, zipfile.BadZipFile) as error:
        raise KnowledgeBaseError(f"архив базы не читается: {path.name}") from error
    if not files:
        raise KnowledgeBaseError(f"в архиве базы нет текстовых файлов: {path.name}")
    return tuple(sorted(files, key=lambda file: file.name))


def company_document(document_id: DocumentId, source: SourceFile) -> Document:
    header = read_header(source.text)
    if not header.title:
        # В тексте нет шапки: называем документ по файлу, иначе на него не сослаться.
        header = replace(header, title=PurePosixPath(source.name).stem)
    return Document(
        id=document_id, header=header, provisions=parse(source.text).provisions
    )


def _is_text_file(info: zipfile.ZipInfo) -> bool:
    name = PurePosixPath(info.filename)
    if info.is_dir() or name.suffix.lower() not in _TEXT_SUFFIXES:
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
