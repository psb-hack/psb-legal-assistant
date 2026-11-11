"""Сборка ZIP-сабмита: белый список, сортировка имён, фиксированная дата."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# Дата в ZipInfo фиксирована, чтобы одинаковый вход давал одинаковый хеш.
_ZIP_EPOCH = (2026, 1, 1, 0, 0, 0)

SRC_MODULES = (
    "solution.py",
    "lex_errors.py",
    "lex_secrets.py",
    "lex_settings.py",
    "lex_submission.py",
)


def build(repo_root: Path, dest: Path) -> Path:
    src = repo_root / "src"
    requirements = repo_root / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    dest.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, Path]] = [
        ("requirements.txt", requirements),
    ]
    for name in SRC_MODULES:
        path = src / name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append((name, path))
    entries.sort(key=lambda item: item[0])
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in entries:
            info = zipfile.ZipInfo(arcname, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
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
    path = build(args.root, dest)
    print(path)


if __name__ == "__main__":
    main()
