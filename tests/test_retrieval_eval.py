"""Разбор источников golden-набора и метрики поиска на выдуманной базе."""

from __future__ import annotations

import pytest

from lex_documents import ProvisionId
from lex_entity_catalog import build_catalog
from lex_knowledge_base import KnowledgeBase
from lex_retrieval import RankedProvision
from lex_sanitizer import AdmittedProvision
from retrieval_eval import (
    EvalCase,
    evaluate,
    found_within,
    load_cases,
    mean_reciprocal_rank,
    resolve_source,
    summary,
)
from synthetic_base import CONTRACT
from synthetic_counterparties import TEXTS, knowledge_base

# К synthetic_counterparties добавлен договор с несколькими пунктами — d11.
BASE = knowledge_base((*TEXTS, CONTRACT))
CATALOG = build_catalog(BASE)
SUPPLY, AMENDMENT, MEMO, REGULATION, LAW, SERVICES = (
    "d01",
    "d03",
    "d08",
    "d09",
    "d10",
    "d11",
)


def _ids(base: KnowledgeBase, document_id: str, *labels: str) -> frozenset[str]:
    document = next(item for item in base.documents if item.id == document_id)
    return frozenset(
        document.provision_id(provision)
        for provision in document.provisions
        if not labels or provision.label in labels
    )


class _FixedRetriever:
    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking

    def retrieve(self, query: str, limit: int) -> tuple[RankedProvision, ...]:
        return tuple(
            RankedProvision(AdmittedProvision(ProvisionId(item), "", ""), 1.0)
            for item in self._ranking[:limit]
        )


def _case(case_id: str, *sources: str, scenario: str = "direct") -> EvalCase:
    return EvalCase(case_id, scenario, "вопрос", sources)


def test_contract_clauses_are_separate_targets() -> None:
    targets = resolve_source(
        "Договор оказания услуг № У-03/2031, п. 2.1, 3.2", BASE, CATALOG
    )
    assert targets == (_ids(BASE, SERVICES, "п. 2.1"), _ids(BASE, SERVICES, "п. 3.2"))
    assert all(len(target) == 1 for target in targets or ())


@pytest.mark.parametrize(
    ("source", "document", "labels"),
    [
        ("Договор поставки № Д-21/2031, п. 1.1", SUPPLY, ("п. 1.1",)),
        ("Доп. соглашение № 1 к Д-21/2031", AMENDMENT, ()),
        ("Служебная записка № ОК-4 от 20.04.2031", MEMO, ()),
        ("Регламент ред. 01.02.2031, п. 1.1", REGULATION, ("п. 1.1",)),
        ("ТК РФ, ст. 300", LAW, ()),
        ("ТК РФ, ст. 300, ч. 2", LAW, ("ч. 2",)),
    ],
)
def test_source_resolves_to_provisions_of_its_document(
    source: str, document: str, labels: tuple[str, ...]
) -> None:
    expected = _ids(BASE, document, *labels)
    assert expected
    assert resolve_source(source, BASE, CATALOG) == (expected,)


@pytest.mark.parametrize(
    "source",
    [
        "Договор поставки № Х-99/2000, п. 1.1",
        "Договор поставки № Д-21/2031, п. 9.9",
        "Договор поставки № Д-21/2031, раздел о цене",
        # Договоров перевозки два, номер не назван.
        "Договор перевозки",
        "Регламент ред. 01.02.2020, п. 1.1",
        "ТК РФ, ст. 301",
        "ТК РФ",
        "ТК РФ, ст. 300, ч. 9",
        "Приказ о премировании",
    ],
)
def test_source_that_does_not_name_provisions_of_the_base_is_unresolved(
    source: str,
) -> None:
    assert resolve_source(source, BASE, CATALOG) is None


def test_outcome_keeps_rank_of_first_and_of_last_source() -> None:
    (paid,) = _ids(BASE, SERVICES, "п. 2.1")
    (claims,) = _ids(BASE, SERVICES, "п. 3.2")
    (subject,) = _ids(BASE, SUPPLY, "п. 1.1")
    cases = [_case("team-01", "Договор оказания услуг № У-03/2031, п. 2.1, 3.2")]
    retriever = _FixedRetriever([subject, claims, subject, paid])
    (outcome,) = evaluate(cases, retriever, BASE, CATALOG).outcomes
    assert outcome.target_ranks == (4, 2)
    assert outcome.first_rank == 2
    assert outcome.complete_rank == 4


def test_source_missing_from_ranking_leaves_the_case_incomplete() -> None:
    (claims,) = _ids(BASE, SERVICES, "п. 3.2")
    cases = [_case("team-01", "Договор оказания услуг № У-03/2031, п. 2.1, 3.2")]
    (outcome,) = evaluate(cases, _FixedRetriever([claims]), BASE, CATALOG).outcomes
    assert outcome.first_rank == 1
    assert outcome.complete_rank is None


def test_cases_without_sources_and_with_unresolved_ones_are_counted_apart() -> None:
    cases = [
        _case("team-01", scenario="off_topic"),
        _case("team-02", "ТК РФ, ст. 300", "Приказ о премировании"),
        _case("demo-03", "ТК РФ, ст. 300"),
    ]
    law = sorted(_ids(BASE, LAW))
    report = summary(evaluate(cases, _FixedRetriever(["d01.1", *law]), BASE, CATALOG))
    assert report["cases"] == {
        "evaluated": 1,
        "without_sources": ["team-01"],
        "unresolved_sources": {"team-02": ["Приказ о премировании"]},
    }
    assert report["overall"] == {
        "cases": 1,
        "any_source": {5: 1, 10: 1, 20: 1},
        "all_sources": {5: 1, 10: 1, 20: 1},
    }
    assert report["mrr"] == 0.5
    assert report["by_origin"] == {"demo": report["overall"]}
    assert report["ranks"] == {"demo-03": {"first": 2, "all": 2}}


def test_found_within_counts_ranks_up_to_depth() -> None:
    assert found_within([1, 5, 6, None], 5) == 2


def test_mean_reciprocal_rank_counts_missing_source_as_zero() -> None:
    assert mean_reciprocal_rank([1, 4, None, None]) == pytest.approx(1.25 / 4)
    assert mean_reciprocal_rank([]) == 0.0


def test_golden_cases_are_read_with_their_sources() -> None:
    cases = load_cases()
    assert len({case.id for case in cases}) == len(cases)
    assert {case.origin for case in cases} == {"demo", "team"}
    assert any(not case.sources for case in cases)
    assert all(case.question for case in cases)
