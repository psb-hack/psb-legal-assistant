from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from lex_dense_retrieval import DenseRetriever, embed_provisions
from lex_errors import RetrievalError
from lex_retrieval import indexed_text
from synthetic_provisions import PREPAYMENT, PROVISIONS
from table_embedder import FailingEmbedder, TableEmbedder


class _SameVectorEmbedder:
    def __init__(self, vectors: Sequence[tuple[float, ...]]) -> None:
        self._vectors = tuple(vectors)

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._vectors[: len(texts)]


def test_provisions_are_embedded_once_in_batches() -> None:
    embedder = TableEmbedder()
    index = embed_provisions(PROVISIONS, embedder, batch_size=3)
    assert [len(call) for call in embedder.calls] == [3, 1]
    sent = [text for call in embedder.calls for text in call]
    assert sent == [indexed_text(item) for item in PROVISIONS]
    assert index.provisions == PROVISIONS
    assert index.vectors.shape == (4, 3)


def test_query_costs_one_embedder_call_and_goes_without_instruction() -> None:
    embedder = TableEmbedder()
    retriever = DenseRetriever(embed_provisions(PROVISIONS, embedder, 12), embedder)
    embedder.calls.clear()
    retriever.retrieve("Можно ли платить авансом?", 2)
    assert embedder.calls == [("Можно ли платить авансом?",)]


def test_paraphrase_without_common_words_finds_the_provision() -> None:
    embedder = TableEmbedder()
    retriever = DenseRetriever(embed_provisions(PROVISIONS, embedder, 12), embedder)
    best = retriever.retrieve("Можно ли платить авансом?", 1)
    assert [hit.provision for hit in best] == [PREPAYMENT]
    assert best[0].score == pytest.approx(1.0)


def test_score_is_cosine_whatever_the_vector_length() -> None:
    long_and_short = _SameVectorEmbedder([(3.0, 4.0), (0.6, 0.8), (0.0, 0.0)])
    index = embed_provisions(PROVISIONS[:3], long_and_short, 12)
    assert np.linalg.norm(index.vectors, axis=1) == pytest.approx([1.0, 1.0, 0.0])
    query = _SameVectorEmbedder([(30.0, 40.0)])
    hits = DenseRetriever(index, query).retrieve("запрос", 3)
    assert [hit.score for hit in hits] == pytest.approx([1.0, 1.0, 0.0])


def test_equal_scores_keep_the_order_of_the_base() -> None:
    same = _SameVectorEmbedder([(1.0, 0.0)] * 4)
    hits = DenseRetriever(embed_provisions(PROVISIONS, same, 12), same).retrieve("x", 3)
    assert [hit.provision for hit in hits] == list(PROVISIONS[:3])


def test_embedder_failure_while_indexing_is_a_retrieval_error() -> None:
    with pytest.raises(RetrievalError, match="эмбеддер не ответил: прокси вернул 503"):
        embed_provisions(PROVISIONS, FailingEmbedder(), 12)


def test_embedder_failure_on_query_is_a_retrieval_error() -> None:
    index = embed_provisions(PROVISIONS, TableEmbedder(), 12)
    with pytest.raises(RetrievalError, match="эмбеддер не ответил"):
        DenseRetriever(index, FailingEmbedder()).retrieve("аванс", 3)


def test_wrong_number_of_vectors_is_a_retrieval_error() -> None:
    with pytest.raises(RetrievalError, match="1 векторов на 4 текстов"):
        embed_provisions(PROVISIONS, _SameVectorEmbedder([(1.0, 0.0)]), 12)


def test_vectors_of_different_size_are_a_retrieval_error() -> None:
    ragged = _SameVectorEmbedder([(1.0, 0.0), (1.0, 0.0, 0.0)])
    with pytest.raises(RetrievalError, match="разной размерности"):
        embed_provisions(PROVISIONS[:2], ragged, 12)


def test_query_vector_of_another_size_is_a_retrieval_error() -> None:
    index = embed_provisions(PROVISIONS, TableEmbedder(), 12)
    other_model = _SameVectorEmbedder([(1.0, 0.0)])
    with pytest.raises(RetrievalError, match="2 и 3"):
        DenseRetriever(index, other_model).retrieve("аванс", 3)


def test_empty_query_returns_nothing_and_does_not_call_the_embedder() -> None:
    embedder = TableEmbedder()
    retriever = DenseRetriever(embed_provisions(PROVISIONS, embedder, 12), embedder)
    embedder.calls.clear()
    assert retriever.retrieve("  ", 3) == ()
    assert embedder.calls == []


def test_empty_base_returns_nothing_and_does_not_call_the_embedder() -> None:
    embedder = TableEmbedder()
    retriever = DenseRetriever(embed_provisions((), embedder, 12), embedder)
    assert retriever.retrieve("аванс", 3) == ()
    assert embedder.calls == []
