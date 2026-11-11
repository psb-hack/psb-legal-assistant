"""Протоколы модели и типы ответов прокси. Без httpx."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    finish_reason: str


@dataclass(frozen=True, slots=True)
class RankedDocument:
    index: int
    score: float


class ChatModel(Protocol):
    def complete(self, turns: Sequence[ChatTurn]) -> ChatResult: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str]
    ) -> tuple[RankedDocument, ...]: ...
