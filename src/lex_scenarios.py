"""Сценарии ответа из docs/03-scenarios.md.

Значения совпадают с метками golden-набора (eval/README.md): классификатор,
модель и оценка называют сценарий одним словом.
"""

from __future__ import annotations

from enum import Enum


class Scenario(Enum):
    DIRECT = "direct"
    INSUFFICIENT = "insufficient"
    FALSE_PREMISE = "false_premise"
    EDITION = "edition"
    # Просьба помочь с противоправным действием, а не вопрос «законно ли это».
    ILLEGAL = "illegal"
    ROLE_ATTACK = "role_attack"
    DIALOG = "dialog"
    AMBIGUOUS = "ambiguous"
    OFF_TOPIC = "off_topic"
