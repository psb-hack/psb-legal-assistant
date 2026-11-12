"""Оценка поиска по golden-набору: разбор поля sources, recall@k и MRR.

Источник кейса — строка для человека: «Договор аренды № А-08/2024, п. 3.1, 3.2»,
«ТК РФ, ст. 70», «Регламент ред. 01.03.2025, п. 2.4». Здесь она переводится в
идентификаторы положений базы. Источник, который перевести не удалось, из
подсчёта не исчезает: его кейс попадает в отдельный список.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from lex_document_numbers import DocumentNumber, find_document_numbers
from lex_documents import (
    ClauseNumber,
    Document,
    DocumentKind,
    LawAddress,
    LawArticleHeader,
    LawUnit,
    LawUnitKind,
    ProvisionId,
)
from lex_entity_catalog import DocumentEntry, EntityCatalog
from lex_knowledge_base import KnowledgeBase
from lex_normalization import comparison_key
from lex_retrieval import Retriever

CASES = Path(__file__).resolve().parents[1] / "eval" / "cases.yaml"
RECALL_DEPTHS = (5, 10, 20)

# Положения, любое из которых закрывает одну названную в источнике единицу: один
# пункт договора — одно положение, статья закона без части — все её положения.
Target = frozenset[ProvisionId]

# «ст. 70», «ст. 13.11»
_ARTICLE = re.compile(r"ст\.\s*(?P<article>\d+(?:\.\d+)*)")
# «ч. 2», «подп. 2.1», «п. 1.8-3»
_LAW_UNIT = re.compile(r"(?P<kind>ч\.|п\.|подп\.|прим\.)\s*(?P<number>\d[\d.\-]*)")
# «п. 3.1» и продолжение перечисления «3.2»
_CLAUSE = re.compile(r"(?:п\.\s*)?(?P<number>\d+(?:\.\d+)*)")
# «ред. 01.03.2025»
_EDITION = re.compile(r"ред\.\s*(?P<date>\d{2}\.\d{2}\.\d{4})")
_EDITION_FORMAT = "%d.%m.%Y"
# Первое слово источника в ключе сравнения. «Доп. соглашение № 1 к П-17/2024»
# содержит номер договора, поэтому вид документа нужен, чтобы их различить.
_KIND_BY_FIRST_WORDS = (
    ("доп", DocumentKind.SUPPLEMENTARY_AGREEMENT),
    ("договор", DocumentKind.CONTRACT),
    ("регламент", DocumentKind.REGULATION),
    ("письмо", DocumentKind.LETTER),
    ("служебная записка", DocumentKind.MEMO),
)


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    scenario: str
    question: str
    sources: tuple[str, ...]

    @property
    def origin(self) -> str:
        """«demo» — вопрос организаторов, «team» — кейс команды."""
        return self.id.split("-")[0]


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case: EvalCase
    # Место первого положения каждой цели в выдаче, с единицы; None — не найдено.
    target_ranks: tuple[int | None, ...]

    @property
    def first_rank(self) -> int | None:
        """Место, на котором в выдаче появился хотя бы один источник."""
        found = [rank for rank in self.target_ranks if rank is not None]
        return min(found, default=None)

    @property
    def complete_rank(self) -> int | None:
        """Место, к которому в выдаче собрались все источники кейса."""
        if None in self.target_ranks:
            return None
        return max(rank for rank in self.target_ranks if rank is not None)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    outcomes: tuple[CaseOutcome, ...]
    # Кейсы, где искать нечего: ответа в базе нет или вопрос не по профилю.
    without_sources: tuple[EvalCase, ...]
    # Кейс и его источники, которые не удалось перевести в положения базы.
    unresolved: tuple[tuple[EvalCase, tuple[str, ...]], ...]


def load_cases(path: Path = CASES) -> tuple[EvalCase, ...]:
    return tuple(
        EvalCase(
            id=item["id"],
            scenario=item["scenario"],
            question=item["question"].strip(),
            sources=tuple(item["sources"]),
        )
        for item in yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def resolve_source(
    source: str, base: KnowledgeBase, catalog: EntityCatalog
) -> tuple[Target, ...] | None:
    """Цели источника; None — документ или адрес в базе не найден однозначно."""
    name, *addresses = (part.strip() for part in source.split(","))
    act = comparison_key(name)
    articles = [
        document
        for document in base.documents
        if isinstance(document.header, LawArticleHeader)
        and comparison_key(document.header.act_short) == act
    ]
    if articles:
        return _law_targets(articles, addresses)
    document = _company_document(name, base, catalog)
    return None if document is None else _clause_targets(document, addresses)


def evaluate(
    cases: Iterable[EvalCase],
    retriever: Retriever,
    base: KnowledgeBase,
    catalog: EntityCatalog,
) -> RetrievalEvaluation:
    """Запрос — вопрос кейса как есть; диалог уходит в поиск целиком."""
    depth = sum(len(document.provisions) for document in base.documents)
    outcomes: list[CaseOutcome] = []
    without_sources: list[EvalCase] = []
    unresolved: list[tuple[EvalCase, tuple[str, ...]]] = []
    for case in cases:
        if not case.sources:
            without_sources.append(case)
            continue
        resolved = {
            source: resolve_source(source, base, catalog) for source in case.sources
        }
        failed = tuple(
            source for source, targets in resolved.items() if targets is None
        )
        if failed:
            unresolved.append((case, failed))
            continue
        targets = [target for found in resolved.values() for target in found or ()]
        retrieved = [
            hit.provision.id for hit in retriever.retrieve(case.question, depth)
        ]
        outcomes.append(
            CaseOutcome(case, tuple(_rank_of(target, retrieved) for target in targets))
        )
    return RetrievalEvaluation(
        tuple(outcomes), tuple(without_sources), tuple(unresolved)
    )


def found_within(ranks: Iterable[int | None], depth: int) -> int:
    return sum(rank is not None and rank <= depth for rank in ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Не найденный источник даёт ноль; пустой набор — тоже ноль."""
    if not ranks:
        return 0.0
    return sum(0.0 if rank is None else 1 / rank for rank in ranks) / len(ranks)


def summary(evaluation: RetrievalEvaluation) -> dict[str, Any]:
    """Цифры прогона в виде, в котором они лежат в tests/golden."""
    outcomes = evaluation.outcomes
    return {
        "cases": {
            "evaluated": len(outcomes),
            "without_sources": [case.id for case in evaluation.without_sources],
            "unresolved_sources": {
                case.id: list(sources) for case, sources in evaluation.unresolved
            },
        },
        "overall": _counts(outcomes),
        "mrr": round(mean_reciprocal_rank([item.first_rank for item in outcomes]), 3),
        "by_scenario": _grouped(outcomes, lambda item: item.case.scenario),
        "by_origin": _grouped(outcomes, lambda item: item.case.origin),
        "ranks": {
            item.case.id: {"first": item.first_rank, "all": item.complete_rank}
            for item in outcomes
        },
    }


def _counts(outcomes: Sequence[CaseOutcome]) -> dict[str, Any]:
    first = [item.first_rank for item in outcomes]
    complete = [item.complete_rank for item in outcomes]
    return {
        "cases": len(outcomes),
        "any_source": {depth: found_within(first, depth) for depth in RECALL_DEPTHS},
        "all_sources": {
            depth: found_within(complete, depth) for depth in RECALL_DEPTHS
        },
    }


def _grouped(
    outcomes: Sequence[CaseOutcome], key: Callable[[CaseOutcome], str]
) -> dict[str, Any]:
    groups: dict[str, list[CaseOutcome]] = {}
    for item in outcomes:
        groups.setdefault(key(item), []).append(item)
    return {name: _counts(members) for name, members in sorted(groups.items())}


def _rank_of(target: Target, retrieved: Sequence[ProvisionId]) -> int | None:
    places = (
        place for place, found in enumerate(retrieved, start=1) if found in target
    )
    return next(places, None)


def _law_targets(
    articles: Sequence[Document], addresses: Sequence[str]
) -> tuple[Target, ...] | None:
    if not addresses or (named := _ARTICLE.fullmatch(addresses[0])) is None:
        return None
    units: list[LawUnit] = []
    for address in addresses[1:]:
        if (unit := _LAW_UNIT.fullmatch(address)) is None:
            return None
        units.append(LawUnit(LawUnitKind(unit["kind"]), unit["number"]))
    for document in articles:
        header = document.header
        if isinstance(header, LawArticleHeader) and header.article == named["article"]:
            target = frozenset(
                document.provision_id(provision)
                for provision in document.provisions
                if isinstance(provision.number, LawAddress)
                and provision.number.units[: len(units)] == tuple(units)
            )
            return (target,) if target else None
    return None


def _clause_targets(
    document: Document, addresses: Sequence[str]
) -> tuple[Target, ...] | None:
    if not addresses:
        return (frozenset(map(document.provision_id, document.provisions)),)
    targets: list[Target] = []
    for address in addresses:
        if (written := _CLAUSE.fullmatch(address)) is None:
            return None
        clause = ClauseNumber(tuple(int(part) for part in written["number"].split(".")))
        target = frozenset(
            document.provision_id(provision)
            for provision in document.provisions
            if provision.number == clause
        )
        if not target:
            return None
        targets.append(target)
    return tuple(targets)


def _company_document(
    name: str, base: KnowledgeBase, catalog: EntityCatalog
) -> Document | None:
    key = comparison_key(name)
    kinds = [kind for words, kind in _KIND_BY_FIRST_WORDS if key.startswith(words)]
    if not kinds:
        return None
    numbers = [mention.number for mention in find_document_numbers(name)]
    edition = _edition(name)
    candidates = [
        entry
        for entry in catalog.documents
        if entry.kind is kinds[0]
        and (edition is None or entry.edition == edition)
        and all(number in _numbers_of(entry) for number in numbers)
    ]
    if len(candidates) != 1:
        return None
    return next(doc for doc in base.documents if doc.id == candidates[0].id)


def _edition(name: str) -> date | None:
    written = _EDITION.search(name)
    if written is None:
        return None
    return datetime.strptime(written["date"], _EDITION_FORMAT).date()


def _numbers_of(entry: DocumentEntry) -> set[DocumentNumber]:
    # У допсоглашения свой номер — «№ 1», а номер договора стоит в заголовке.
    in_title = {mention.number for mention in find_document_numbers(entry.title)}
    return in_title if entry.number is None else in_title | {entry.number}
