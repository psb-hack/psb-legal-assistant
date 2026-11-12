"""Протоколы модели и типы ответов прокси. Без httpx."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    finish_reason: str


class FinishReason(Enum):
    """Чем закончилась генерация, ответ которой дошёл до вызывающего кода."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    # Прокси прислал значение, которого нет в этом списке, или не прислал ничего.
    UNRECOGNIZED = "unrecognized"


@dataclass(frozen=True, slots=True)
class JsonCompletion:
    # Бывает пустым или оборванным: что с этим делать, решает вызывающий код.
    content: str
    finish: FinishReason


@dataclass(frozen=True, slots=True)
class RankedDocument:
    index: int
    score: float


class ChatModel(Protocol):
    def complete(self, turns: Sequence[ChatTurn]) -> ChatResult: ...


class JsonChatModel(Protocol):
    def complete_json(self, turns: Sequence[ChatTurn]) -> JsonCompletion: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str]
    ) -> tuple[RankedDocument, ...]: ...
