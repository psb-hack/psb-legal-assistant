"""Маршрут на настоящей базе: планы кейсов golden-набора, источник в контексте
и цена маршрута в знаках. Архив в репозиторий не входит."""

from __future__ import annotations

import functools
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_dates import PartialDate
from lex_dialogue import (
    DialogueEntities,
    Question,
    dialogue_entities,
    dialogue_event_date,
    last_question,
    parse_dialogue,
    search_query,
)
from lex_document_graph import DocumentGraph, build_graph
from lex_documents import DocumentHeader, DocumentId, LawArticleHeader
from lex_edition_choice import EditionInForce
from lex_entity_catalog import EntityCatalog, build_catalog
from lex_entity_recognition import Evidence, Mention, Resolved
from lex_event_date import EventDate
from lex_knowledge_base import KnowledgeBase, load_knowledge_base
from lex_retrieval import LexicalRetriever, RankedProvision
from lex_routing import (
    CONTEXT_CHARACTER_LIMIT,
    LONG_DOCUMENT_CHARACTERS,
    ContextLimits,
    RetrievalPlan,
    RetrievedContext,
    RouteReason,
    admitted_documents,
    assemble_context,
    own_company,
    plan_retrieval,
)
from lex_sanitizer import AdmittedProvision, default_detectors, screen
from lex_settings import SanitizerThresholds
from retrieval_eval import (
    CaseOutcome,
    EvalCase,
    evaluate,
    load_cases,
    resolve_source,
)

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "routing.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
TODAY = EXPECTED["today"]
LIMITS = ContextLimits()
DEMO_DIALOGUES = ("demo-06", "demo-15", "demo-16", "demo-17", "demo-18")

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


@functools.cache
def _base() -> KnowledgeBase:
    return load_knowledge_base(ARCHIVE)


@functools.cache
def _catalog() -> EntityCatalog:
    return build_catalog(_base())


@functools.cache
def _graph() -> DocumentGraph:
    return build_graph(_base())


@functools.cache
def _documents() -> dict[DocumentId, tuple[AdmittedProvision, ...]]:
    thresholds = SanitizerThresholds()
    screening = screen(_base(), default_detectors(thresholds), thresholds)
    return admitted_documents(_base(), screening)


@functools.cache
def _retriever() -> LexicalRetriever:
    return LexicalRetriever(
        [provision for own in _documents().values() for provision in own]
    )


def _cases() -> dict[str, EvalCase]:
    return {case.id: case for case in load_cases()}


def _question(text: str) -> Question:
    question = last_question(parse_dialogue(text))
    assert isinstance(question, Question)
    return question


def _plan(text: str) -> RetrievalPlan:
    question = _question(text)
    return plan_retrieval(
        dialogue_entities(question, _catalog()),
        dialogue_event_date(question, TODAY),
        TODAY,
        _catalog(),
        _graph(),
        _base(),
    )


def _ranking(text: str) -> tuple[RankedProvision, ...]:
    question = _question(text)
    query = search_query(question, dialogue_entities(question, _catalog()))
    depth = sum(len(own) for own in _documents().values())
    return _retriever().retrieve(query, depth)


def _context(text: str, plan: RetrievalPlan) -> RetrievedContext:
    return assemble_context(plan, _documents(), _ranking(text), LIMITS)


def _routed(text: str) -> RetrievedContext:
    return _context(text, _plan(text))


def _searched(text: str) -> RetrievedContext:
    """Контекст без маршрута: одни лучшие положения общего поиска."""
    return _context(text, RetrievalPlan((), (), (), (), None))


def _title(document: DocumentId) -> str:
    return next(
        item.header.citation_title for item in _base().documents if item.id == document
    )


def _plan_reading(plan: RetrievalPlan) -> dict[str, Any]:
    return {
        "routes": [
            [_title(route.document), route.coverage.value]
            + [reason.name.lower() for reason in route.reasons]
            for route in plan.routes
        ],
        "ambiguous": [mention.text for mention in plan.ambiguous],
        "not_found": [mention.text for mention in plan.not_found],
        "own_company": None if plan.own_company is None else plan.own_company.name,
    }


def _size(context: RetrievedContext) -> int:
    return sum(len(item.heading) + len(item.text) for item in context.admitted)


def _outcomes(assembled: Any) -> tuple[CaseOutcome, ...]:
    def search(text: str, depth: int) -> Sequence[RankedProvision]:
        return [RankedProvision(item, 0.0) for item in assembled(text).admitted]

    return evaluate(load_cases(), search, _base(), _catalog()).outcomes


def _shares(outcomes: Sequence[CaseOutcome]) -> dict[str, str]:
    return {
        "any_source": f"{sum(item.first_rank is not None for item in outcomes)}"
        f" из {len(outcomes)}",
        "all_sources": f"{sum(item.complete_rank is not None for item in outcomes)}"
        f" из {len(outcomes)}",
    }


def _coverage(assembled: Any) -> dict[str, Any]:
    outcomes = _outcomes(assembled)
    contexts = [assembled(case.question) for case in load_cases()]
    counts = [len(context.provisions) for context in contexts]
    sizes = [_size(context) for context in contexts]
    scenarios = sorted({item.case.scenario for item in outcomes})
    return {
        "overall": _shares(outcomes),
        "by_scenario": {
            scenario: _shares(
                [item for item in outcomes if item.case.scenario == scenario]
            )
            for scenario in scenarios
        },
        "missing_any": [item.case.id for item in outcomes if item.first_rank is None],
        "missing_some": [
            item.case.id
            for item in outcomes
            if item.first_rank is not None and item.complete_rank is None
        ],
        "context": {
            "provisions": {
                "median": statistics.median(counts),
                "max": max(counts),
            },
            "characters": {"median": statistics.median(sizes), "max": max(sizes)},
        },
    }


def _source_reasons(case: EvalCase) -> list[tuple[RouteReason, ...] | None]:
    """По каждому источнику кейса — причины его положения в контексте; None —
    источника в контексте нет."""
    context = _routed(case.question)
    reasons = {item.provision.id: item.reasons for item in context.provisions}
    found: list[tuple[RouteReason, ...] | None] = []
    for source in case.sources:
        for target in resolve_source(source, _base(), _catalog()) or ():
            inside = [
                reasons[provision] for provision in reasons if provision in target
            ]
            found.append(inside[0] if inside else None)
    return found


def test_plans_of_golden_cases() -> None:
    plans = {case.id: _plan(case.question) for case in load_cases()}
    readings = {
        case_id: _plan_reading(plan)
        for case_id, plan in plans.items()
        if plan != RetrievalPlan((), (), (), (), None)
    }
    assert readings == EXPECTED["plans"]


@pytest.mark.parametrize("case_id", DEMO_DIALOGUES)
def test_demo_dialogue_gets_its_source_and_the_reason(case_id: str) -> None:
    reasons = _source_reasons(_cases()[case_id])
    assert reasons
    assert None not in reasons
    expected = EXPECTED["demo_dialogues"][case_id]
    assert [[reason.name.lower() for reason in found or ()] for found in reasons] == (
        expected
    )


def test_dialogue_without_entities_is_answered_by_search() -> None:
    # Демо 18: ассистент назвал только свою компанию, регламент не назван никем.
    plan = _plan(_cases()["demo-18"].question)
    assert plan.routes == ()
    assert plan.own_company == own_company(_catalog())


def test_contract_number_brings_the_supplementary_agreement() -> None:
    question = "Какой срок оплаты по договору П-17/2024?"
    contract, agreement = _plan(question).routes
    assert _title(contract.document) == "Договор поставки № П-17/2024"
    assert contract.reasons == (RouteReason.NAMED_BY_USER,)
    assert "Дополнительное соглашение № 1" in _title(agreement.document)
    assert agreement.reasons == (RouteReason.LINKED,)
    context = _routed(question)
    inside = {item.provision.id for item in context.provisions}
    assert {item.id for item in _documents()[agreement.document]} <= inside
    assert set(contract.required) | set(agreement.required) <= inside
    assert len(contract.required) == 1
    assert len(agreement.required) == EXPECTED["agreement_required"]


@pytest.mark.parametrize("case_id", ["demo-15", "team-06", "team-01"])
def test_named_contract_puts_every_source_into_the_context(case_id: str) -> None:
    case = _cases()[case_id]
    assert None not in _source_reasons(case)
    searched = {item.id for item in _searched(case.question).admitted}
    targets = [
        target
        for source in case.sources
        for target in resolve_source(source, _base(), _catalog()) or ()
    ]
    missed = [target for target in targets if not target & searched]
    assert len(missed) == EXPECTED["missed_by_search"][case_id]


def test_regulation_edition_follows_the_event_date() -> None:
    # Регламент по названию пока не распознаётся: сущность подставлена вручную.
    current, earlier = (
        entry
        for date_ in sorted(EXPECTED["regulation_editions"], reverse=True)
        for entry in _catalog().documents
        if entry.edition == date_
    )
    named = DialogueEntities(
        (Mention("регламент", Resolved(current, Evidence.FULL_NAME)),), ()
    )
    asked = EventDate(PartialDate.of_month(2024, 8), "в августе 2024")
    plan = plan_retrieval(named, asked, TODAY, _catalog(), _graph(), _base())
    (route,) = plan.routes
    assert route.document == earlier.id
    assert route.reasons == (
        RouteReason.NAMED_BY_USER,
        RouteReason.EDITION_BY_EVENT_DATE,
    )
    assert isinstance(plan.editions[0].choice, EditionInForce)
    undated = plan_retrieval(named, None, TODAY, _catalog(), _graph(), _base())
    assert [_title(item.document) for item in undated.routes] == (
        EXPECTED["regulation_without_date"]
    )


def test_own_company_of_the_base() -> None:
    own = own_company(_catalog())
    assert own is not None
    assert own.name == EXPECTED["own_company"]


def test_long_document_threshold_separates_company_documents_from_laws() -> None:
    lengths = {
        document.id: sum(len(provision.text) for provision in document.provisions)
        for document in _base().documents
    }
    company = [
        lengths[item.id]
        for item in _base().documents
        if isinstance(item.header, DocumentHeader)
    ]
    laws = [
        lengths[item.id]
        for item in _base().documents
        if isinstance(item.header, LawArticleHeader)
    ]
    assert {
        "longest_company_document": max(company),
        "law_articles": [min(laws), max(laws)],
        "long_law_articles": sum(size > LONG_DOCUMENT_CHARACTERS for size in laws),
    } == EXPECTED["document_lengths"]
    assert max(company) < LONG_DOCUMENT_CHARACTERS


def test_no_golden_case_reaches_the_context_limit() -> None:
    for case in load_cases():
        context = _routed(case.question)
        assert context.reduced == (), case.id
        assert _size(context) <= CONTEXT_CHARACTER_LIMIT


def test_source_coverage_with_and_without_the_route() -> None:
    assert _coverage(_searched) == EXPECTED["coverage"]["search_only"]
    assert _coverage(_routed) == EXPECTED["coverage"]["with_route"]
