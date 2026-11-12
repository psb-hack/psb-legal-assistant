"""Векторный поиск положений: эмбеддинги за протоколом Embedder, точный перебор.

База — сотни положений, поэтому приближённый индекс не нужен: запрос — одно
умножение матрицы на вектор.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lex_errors import ModelGatewayError, RetrievalError
from lex_models import Embedder
from lex_retrieval import RankedProvision, indexed_text, rank
from lex_sanitizer import AdmittedProvision

# Размер батча из примера в карточке bge-m3. Лимит прокси на число и длину
# текстов в запросе неизвестен: это значение — ориентир до первого прогона.
EMBEDDING_BATCH_SIZE = 12


# eq=False: сравнение по полям упёрлось бы в поэлементное сравнение матриц.
@dataclass(frozen=True, slots=True, eq=False)
class DenseIndex:
    provisions: tuple[AdmittedProvision, ...]
    # Строка i — вектор положения i единичной длины.
    vectors: NDArray[np.float64]


def embed_provisions(
    provisions: Sequence[AdmittedProvision], embedder: Embedder, batch_size: int
) -> DenseIndex:
    """Эмбеддинги всех положений; считаются один раз, при индексации.

    Текст тот же, что у лексического индекса: заголовок фрагмента и положение.
    """
    if not provisions:
        return DenseIndex((), np.empty((0, 0), dtype=np.float64))
    rows: list[tuple[float, ...]] = []
    for start in range(0, len(provisions), batch_size):
        batch = provisions[start : start + batch_size]
        rows.extend(_embedded([indexed_text(item) for item in batch], embedder))
    if len({len(row) for row in rows}) > 1:
        raise RetrievalError("эмбеддер вернул векторы разной размерности")
    return DenseIndex(tuple(provisions), _unit_length(np.array(rows, dtype=np.float64)))


class DenseRetriever:
    """Скалярное произведение вектора запроса и векторов положений."""

    def __init__(self, index: DenseIndex, embedder: Embedder) -> None:
        self._index = index
        self._embedder = embedder

    def retrieve(self, query: str, limit: int) -> tuple[RankedProvision, ...]:
        """Один вызов эмбеддера на запрос. RetrievalError — эмбеддер не ответил."""
        if not query.strip() or not self._index.provisions:
            return ()
        # Запрос уходит без инструкции-префикса: bge-m3 она не нужна.
        (row,) = _embedded([query], self._embedder)
        vector = _unit_length(np.array([row], dtype=np.float64))[0]
        vectors = self._index.vectors
        if vector.shape[0] != vectors.shape[1]:
            raise RetrievalError(
                "размерность вектора запроса не совпала с индексом: "
                f"{vector.shape[0]} и {vectors.shape[1]}"
            )
        scores = [float(score) for score in vectors @ vector]
        return rank(self._index.provisions, scores, limit)


def _embedded(
    texts: Sequence[str], embedder: Embedder
) -> tuple[tuple[float, ...], ...]:
    try:
        vectors = embedder.embed(texts)
    except ModelGatewayError as error:
        raise RetrievalError(f"эмбеддер не ответил: {error}") from error
    if len(vectors) != len(texts):
        raise RetrievalError(
            f"эмбеддер вернул {len(vectors)} векторов на {len(texts)} текстов"
        )
    return vectors


def _unit_length(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Протокол Embedder длину вектора не обещает, а без неё произведение не косинус."""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    # Нулевой вектор остаётся нулевым: его оценка — ноль при любом запросе.
    unit: NDArray[np.float64] = rows / np.where(norms == 0, 1.0, norms)
    return unit
