from __future__ import annotations

import pytest

from lex_dense_retrieval import DenseRetriever, embed_provisions
from lex_documents import ProvisionId
from lex_errors import RetrievalError
from lex_hybrid_retrieval import (
    FUSION_DEPTH,
    RRF_RANK_CONSTANT,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from lex_retrieval import LexicalRetriever, RankedProvision
from lex_sanitizer import AdmittedProvision
from synthetic_provisions import PREPAYMENT, PROVISIONS, RENT, SUBLEASE, WARRANTY
from table_embedder import FailingEmbedder, TableEmbedder


def _hits(*provisions: AdmittedProvision) -> tuple[RankedProvision, ...]:
    return tuple(
        RankedProvision(provision, float(len(provisions) - place))
        for place, provision in enumerate(provisions)
    )


class _FixedRetriever:
    def __init__(self, hits: tuple[RankedProvision, ...]) -> None:
        self._hits = hits
        self.limits: list[int] = []

    def retrieve(self, query: str, limit: int) -> tuple[RankedProvision, ...]:
        self.limits.append(limit)
        return self._hits[:limit]


class _BrokenRetriever:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def retrieve(self, query: str, limit: int) -> tuple[RankedProvision, ...]:
        raise self._error


def test_single_ranking_keeps_its_order_and_gets_reciprocal_scores() -> None:
    fused = reciprocal_rank_fusion([_hits(WARRANTY, RENT, PREPAYMENT)])
    assert [hit.provision for hit in fused] == [WARRANTY, RENT, PREPAYMENT]
    assert [hit.score for hit in fused] == pytest.approx(
        [1 / (RRF_RANK_CONSTANT + place) for place in (1, 2, 3)]
    )


def test_provision_found_by_both_searches_outranks_the_top_of_either() -> None:
    lexical = _hits(RENT, PREPAYMENT, WARRANTY)
    dense = _hits(SUBLEASE, PREPAYMENT)
    fused = reciprocal_rank_fusion([lexical, dense])
    assert [hit.provision for hit in fused] == [PREPAYMENT, RENT, SUBLEASE, WARRANTY]
    assert fused[0].score == pytest.approx(2 / (RRF_RANK_CONSTANT + 2))


def test_original_scores_do_not_affect_the_fusion() -> None:
    lexical = (RankedProvision(RENT, 40.0), RankedProvision(WARRANTY, 0.1))
    dense = (RankedProvision(WARRANTY, 0.9), RankedProvision(SUBLEASE, 0.8))
    fused = reciprocal_rank_fusion([lexical, dense])
    assert [hit.provision for hit in fused] == [WARRANTY, RENT, SUBLEASE]


def test_equal_sums_are_ordered_by_the_earlier_ranking() -> None:
    lexical, dense = _hits(RENT, WARRANTY), _hits(SUBLEASE, PREPAYMENT)
    fused = reciprocal_rank_fusion([lexical, dense])
    assert [hit.provision for hit in fused] == [RENT, SUBLEASE, WARRANTY, PREPAYMENT]
    assert fused[0].score == fused[1].score
    swapped = reciprocal_rank_fusion([dense, lexical])
    assert [hit.provision for hit in swapped] == [SUBLEASE, RENT, PREPAYMENT, WARRANTY]


def test_fusion_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion([]) == ()
    assert reciprocal_rank_fusion([(), ()]) == ()


def test_hybrid_fuses_every_search_and_reports_nothing_unavailable() -> None:
    lexical = _FixedRetriever(_hits(RENT, PREPAYMENT))
    dense = _FixedRetriever(_hits(SUBLEASE, PREPAYMENT))
    found = HybridRetriever((lexical, dense), reciprocal_rank_fusion).retrieve("q", 2)
    assert [hit.provision for hit in found.hits] == [PREPAYMENT, RENT]
    assert found.unavailable == ()


def test_each_search_is_asked_for_the_same_depth_whatever_the_limit() -> None:
    lexical, dense = _FixedRetriever(_hits(RENT)), _FixedRetriever(_hits(SUBLEASE))
    HybridRetriever((lexical, dense), reciprocal_rank_fusion).retrieve("q", 1)
    assert lexical.limits == dense.limits == [FUSION_DEPTH]


def test_without_embedder_hybrid_answers_with_bm25_and_says_so() -> None:
    lexical = LexicalRetriever(PROVISIONS)
    index = embed_provisions(PROVISIONS, TableEmbedder(), 12)
    dense = DenseRetriever(index, FailingEmbedder())
    hybrid = HybridRetriever((lexical, dense), reciprocal_rank_fusion)
    found = hybrid.retrieve("гарантийный срок по договору Д-21/2031", 3)
    alone = lexical.retrieve("гарантийный срок по договору Д-21/2031", 3)
    assert [hit.provision for hit in found.hits] == [hit.provision for hit in alone]
    assert found.hits[0].provision == WARRANTY
    assert found.unavailable == ("эмбеддер не ответил: прокси вернул 503",)


def test_paraphrase_reaches_the_hybrid_through_dense_search() -> None:
    embedder = TableEmbedder()
    dense = DenseRetriever(embed_provisions(PROVISIONS, embedder, 12), embedder)
    lexical = LexicalRetriever(PROVISIONS)
    question = "Можно ли сдать склад другой фирме?"
    assert SUBLEASE not in [hit.provision for hit in lexical.retrieve(question, 4)]
    found = HybridRetriever((lexical, dense), reciprocal_rank_fusion).retrieve(
        question, 4
    )
    assert SUBLEASE in [hit.provision for hit in found.hits]
    assert found.unavailable == ()


def test_every_search_unavailable_gives_empty_hits_and_all_reasons() -> None:
    broken = _BrokenRetriever(RetrievalError("индекс не построен"))
    dense = DenseRetriever(
        embed_provisions(PROVISIONS, TableEmbedder(), 12), FailingEmbedder()
    )
    found = HybridRetriever((broken, dense), reciprocal_rank_fusion).retrieve("q", 3)
    assert found.hits == ()
    assert found.unavailable == (
        "индекс не построен",
        "эмбеддер не ответил: прокси вернул 503",
    )


def test_error_that_is_not_about_retrieval_is_not_swallowed() -> None:
    broken = _BrokenRetriever(ValueError("ошибка в коде"))
    hybrid = HybridRetriever((broken,), reciprocal_rank_fusion)
    with pytest.raises(ValueError, match="ошибка в коде"):
        hybrid.retrieve("q", 3)


def test_fused_ids_stay_unique() -> None:
    fused = reciprocal_rank_fusion([_hits(RENT, WARRANTY), _hits(WARRANTY, RENT)])
    assert sorted(hit.provision.id for hit in fused) == [
        ProvisionId("d01.7"),
        ProvisionId("d04.5"),
    ]
