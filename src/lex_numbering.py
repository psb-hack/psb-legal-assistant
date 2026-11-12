"""Грамматика нумерации положений: «3.1. Текст» в начале строки."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lex_documents import ClauseNumber

# «3.1. Арендная плата составляет…» → номер «3.1», текст после него.
# Часть номера — до трёх цифр без ведущего нуля, поэтому дата в начале строки
# («01.03.2025. Действует…») номером не считается.
_NUMBERED_LINE = re.compile(r"^([1-9]\d{0,2}(?:\.[1-9]\d{0,2})*)\.\s+(\S.*)$")


@dataclass(frozen=True, slots=True)
class NumberedLine:
    number: ClauseNumber
    text: str


def split_numbered_line(line: str) -> NumberedLine | None:
    match = _NUMBERED_LINE.match(line)
    if match is None:
        return None
    parts = tuple(int(part) for part in match.group(1).split("."))
    return NumberedLine(number=ClauseNumber(parts), text=match.group(2))
