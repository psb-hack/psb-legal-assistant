"""Поиск положений по запросу: протокол и лексический поиск BM25.

Индекс строится только из AdmittedProvision: текст в карантине до поиска не
доходит, потому что другого входа у индекса нет.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import bm25s

from lex_sanitizer import AdmittedProvision
from lex_search_tokens import search_tokens

# Значения по умолчанию из литературы. На демо-вопросах k1 от 0,9 до 1,5 и b от
# 0,4 до 0,75 дают разницу в пределах одного положения из двадцати.
BM25_TERM_SATURATION = 1.5
BM25_LENGTH_NORMALIZATION = 0.75
# Вес редкости токена в этом варианте не бывает отрицательным, поэтому нулевая
# оценка означает ровно одно: ни один токен запроса в положении не встретился.
_BM25_VARIANT = "lucene"


@dataclass(frozen=True, slots=True)
class RankedProvision:
    provision: AdmittedProvision
    # Сравнима только внутри одной выдачи одного поиска.
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, limit: int) -> tuple[RankedProvision, ...]:
        """Положения по убыванию оценки, не больше limit.

        RetrievalError — поиск сейчас ответить не может.
        """
        ...


class LexicalRetriever:
    """BM25 по заголовку фрагмента и тексту положения; индекс строится один раз."""

    def __init__(self, provisions: Sequence[AdmittedProvision]) -> None:
        self._provisions = tuple(provisions)
        # Библиотека не строит индекс по пустому набору.
        self._index = _bm25_index(self._provisions) if self._provisions else None

    def retrieve(self, query: str, limit: int) -> tuple[RankedProvision, ...]:
        """Положения без единого общего токена с запросом в выдачу не входят."""
        tokens = search_tokens(query)
        if self._index is None or not tokens:
            return ()
        scores = [float(score) for score in self._index.get_scores(list(tokens))]
        ranked = rank(self._provisions, scores, limit)
        return tuple(hit for hit in ranked if hit.score > 0)


def rank(
    provisions: Sequence[AdmittedProvision], scores: Sequence[float], limit: int
) -> tuple[RankedProvision, ...]:
    """Лучшие по оценке; при равных оценках — в порядке базы.

    Второй ключ делает выдачу одинаковой от запуска к запуску: у положений с
    одинаковым текстом (пункт, не изменившийся между редакциями) оценки равны.
    """
    order = sorted(range(len(provisions)), key=lambda index: (-scores[index], index))
    return tuple(
        RankedProvision(provisions[index], scores[index]) for index in order[:limit]
    )


def indexed_text(provision: AdmittedProvision) -> str:
    """Заголовок фрагмента перед текстом: номер договора, контрагент и редакция
    названы в шапке документа, а не в каждом пункте."""
    return f"{provision.heading}\n{provision.text}"


def _bm25_index(provisions: Sequence[AdmittedProvision]) -> bm25s.BM25:
    index = bm25s.BM25(
        k1=BM25_TERM_SATURATION, b=BM25_LENGTH_NORMALIZATION, method=_BM25_VARIANT
    )
    corpus = [list(search_tokens(indexed_text(item))) for item in provisions]
    index.index(corpus, show_progress=False)
    return index
