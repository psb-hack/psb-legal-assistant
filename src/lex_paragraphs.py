"""Абзацы исходного текста: в документах базы каждая строка файла — абзац."""

from __future__ import annotations

_BYTE_ORDER_MARK = "\ufeff"


def split_paragraphs(text: str) -> list[str]:
    stripped = (
        line.strip() for line in text.removeprefix(_BYTE_ORDER_MARK).splitlines()
    )
    return [line for line in stripped if line]
