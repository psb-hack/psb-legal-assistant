"""Сборка ZIP-сабмита: белый список, сортировка имён, фиксированная дата."""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Дата в ZipInfo фиксирована, чтобы одинаковый вход давал одинаковый хеш.
_ZIP_EPOCH = (2026, 1, 1, 0, 0, 0)

SRC_MODULES = (
    "solution.py",
    "lex_address_detection.py",
    "lex_answer_contract.py",
    "lex_company_parsing.py",
    "lex_dates.py",
    "lex_declension.py",
    "lex_detection.py",
    "lex_document_graph.py",
    "lex_document_header.py",
    "lex_document_links.py",
    "lex_document_numbers.py",
    "lex_documents.py",
    "lex_docx_reading.py",
    "lex_edition_choice.py",
    "lex_entity_catalog.py",
    "lex_entity_recognition.py",
    "lex_errors.py",
    "lex_event_date.py",
    "lex_gateway.py",
    "lex_injection_classifier.py",
    "lex_injection_prompt.py",
    "lex_knowledge_base.py",
    "lex_law_lines.py",
    "lex_law_notes.py",
    "lex_law_parsing.py",
    "lex_law_validity.py",
    "lex_models.py",
    "lex_normalization.py",
    "lex_numbering.py",
    "lex_paragraphs.py",
    "lex_prompt_boundary.py",
    "lex_retry.py",
    "lex_role_markers.py",
    "lex_sanitizer.py",
    "lex_scenarios.py",
    "lex_secrets.py",
    "lex_settings.py",
    "lex_structure_detection.py",
    "lex_submission.py",
)


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    entries: tuple[tuple[str, Path], ...]

    @classmethod
    def default(cls, repo_root: Path) -> ArchiveManifest:
        src = repo_root / "src"
        requirements = repo_root / "requirements.txt"
        if not requirements.is_file():
            raise FileNotFoundError(requirements)
        items: list[tuple[str, Path]] = [("requirements.txt", requirements)]
        for name in SRC_MODULES:
            path = src / name
            if not path.is_file():
                raise FileNotFoundError(path)
            items.append((name, path))
        return cls(tuple(items))


def build(manifest: ArchiveManifest, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(manifest.entries, key=lambda item: item[0])
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in entries:
            if "/" in arcname or "\\" in arcname:
                raise ValueError(f"в архиве только корень: {arcname}")
            info = zipfile.ZipInfo(arcname, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать ZIP-сабмит")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Путь к архиву (по умолчанию dist/submission.zip)",
    )
    args = parser.parse_args()
    dest = args.out if args.out is not None else args.root / "dist" / "submission.zip"
    path = build(ArchiveManifest.default(args.root), dest)
    print(path)


if __name__ == "__main__":
    main()
