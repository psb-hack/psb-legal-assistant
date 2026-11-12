"""Единица проверки санитайзера, сигнал детектора и протокол детектора."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from lex_documents import DocumentId, DocumentPart, HiddenContentKind, ProvisionId
from lex_normalization import NormalizedText


class SignalStrength(Enum):
    NONE = 0
    WEAK = 1
    STRONG = 2


@dataclass(frozen=True, slots=True)
class Signal:
    detector: str
    strength: SignalStrength
    # Признаки, по которым детектор так решил: с ними решение попадает в журнал.
    features: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Fragment:
    """Абзац положения или единица скрытого содержимого документа."""

    document_id: DocumentId
    # None — скрытое содержимое: в положения оно не входит.
    provision_id: ProvisionId | None
    # Номер абзаца внутри положения или номер единицы скрытого содержимого, с нуля.
    line: int
    part: DocumentPart | None
    hidden: HiddenContentKind | None
    # Выше в документе уже стоит подпись, а сам абзац подписью не кончается.
    follows_signature: bool
    text: NormalizedText


class Detector(Protocol):
    def inspect(self, fragment: Fragment) -> Signal: ...
