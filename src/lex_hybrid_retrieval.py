"""Гибридный поиск: слияние выдач нескольких поисков по рангам.

Поиск, который не ответил, из слияния выпадает, а причина остаётся в результате:
без эмбеддера гибрид работает на одном BM25 и говорит об этом.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lex_documents import ProvisionId
from lex_errors import RetrievalError
from lex_retrieval import RankedProvision, Retriever

# Константа из статьи о методе (Cormack и др., 2009): подобрана на слиянии
# десятков систем, а не двух списков, и на нашей базе не проверялась.
RRF_RANK_CONSTANT = 60
# Каждый поиск отдаёт в слияние список одной глубины: положение за её пределами
# получает от поиска ноль, и более длинный список не получает лишних голосов.
FUSION_DEPTH = 50

Fusion = Callable[[Sequence[Sequence[RankedProvision]]], tuple[RankedProvision, ...]]


@dataclass(frozen=True, slots=True)
class HybridRetrieval:
    hits: tuple[RankedProvision, ...]
    # Причины, по которым часть поисков не ответила. Пусто — участвовали все.
    unavailable: tuple[str, ...]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedProvision]],
) -> tuple[RankedProvision, ...]:
    """Оценка положения — сумма 1 / (k + место) по спискам, где оно есть.

    Исходные оценки не участвуют: шкалы BM25 и косинуса несравнимы. При равных
    суммах выше положение, которое встретилось раньше, — списки читаются по
    порядку, поэтому первый поиск имеет приоритет.
    """
    fused: dict[ProvisionId, RankedProvision] = {}
    for ranking in rankings:
        for place, hit in enumerate(ranking, start=1):
            gain = 1 / (RRF_RANK_CONSTANT + place)
            seen = fused.get(hit.provision.id)
            score = gain if seen is None else seen.score + gain
            fused[hit.provision.id] = RankedProvision(hit.provision, score)
    # Сортировка устойчива: порядок первого появления сохраняется при равенстве.
    return tuple(sorted(fused.values(), key=lambda hit: -hit.score))


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], fusion: Fusion) -> None:
        self._retrievers = tuple(retrievers)
        self._fusion = fusion

    def retrieve(self, query: str, limit: int) -> HybridRetrieval:
        rankings: list[tuple[RankedProvision, ...]] = []
        unavailable: list[str] = []
        for retriever in self._retrievers:
            try:
                rankings.append(retriever.retrieve(query, FUSION_DEPTH))
            except RetrievalError as error:
                unavailable.append(str(error))
        return HybridRetrieval(self._fusion(rankings)[:limit], tuple(unavailable))
