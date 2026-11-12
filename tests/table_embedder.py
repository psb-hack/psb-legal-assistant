"""Эмбеддеры для тестов: векторы из выдуманной таблицы и сбой прокси."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lex_errors import ModelGatewayError
from lex_normalization import comparison_key

# Оси: оплата вперёд, гарантия, передача помещения третьим лицам. Слова одной оси
# не имеют общих n-грамм: «аванс» и «предоплата» сближает только эта таблица.
TOPICS: Mapping[str, tuple[float, ...]] = {
    "предоплат": (1.0, 0.0, 0.0),
    "аванс": (1.0, 0.0, 0.0),
    "гарант": (0.0, 1.0, 0.0),
    "субаренд": (0.0, 0.0, 1.0),
    "другой фирме": (0.0, 0.0, 1.0),
}


class TableEmbedder:
    """Вектор текста — сумма строк таблицы, чьи ключи в нём встретились."""

    def __init__(self, topics: Mapping[str, tuple[float, ...]] = TOPICS) -> None:
        self._topics = topics
        self._size = len(next(iter(topics.values())))
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return tuple(self._vector(comparison_key(text)) for text in texts)

    def _vector(self, key: str) -> tuple[float, ...]:
        rows = [row for topic, row in self._topics.items() if topic in key]
        return tuple(sum(row[axis] for row in rows) for axis in range(self._size))


class FailingEmbedder:
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise ModelGatewayError("прокси вернул 503")
