"""Маршрут по сущностям диалога: какие документы идут в контекст и почему.

План — чистая функция от распознанного в диалоге, даты события, каталога и
графа: его можно записать в журнал и показать в интерфейсе до любого поиска.
Исполнение плана собирает контекст из допущенных положений и готовой выдачи
поиска и тоже ничего не вызывает: поиск запускает вызывающий код.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from itertools import zip_longest

from lex_dialogue import DialogueEntities
from lex_document_graph import DocumentGraph, EditionFamily
from lex_document_links import LinkKind
from lex_documents import DocumentId, ProvisionId
from lex_edition_choice import (
    EditionChoice,
    EditionsAcrossBoundary,
    NoEditionOnDate,
    current_edition,
    edition_on,
)
from lex_entity_catalog import Counterparty, EntityCatalog
from lex_entity_recognition import Ambiguous, Entity, Mention, NotFound, Resolved
from lex_event_date import EventDate, UnclearEventDate
from lex_knowledge_base import KnowledgeBase
from lex_retrieval import RankedProvision
from lex_sanitizer import AdmittedProvision, Screening, admit

# Документ длиннее идёт в контекст не целиком, а найденными в нём положениями.
# В базе организаторов самый длинный документ компании — 3,7 тыс. знаков текста
# положений; восемь статей законов из тринадцати — до 6,0 тыс., остальные пять —
# от 10,9 до 84 тыс. Порог стоит в этом разрыве.
LONG_DOCUMENT_CHARACTERS = 8_000
# Глубина общего поиска в контексте и число положений длинного документа. На
# golden-наборе источник есть в первых 10 положениях BM25 у 31 кейса из 36, в
# первых 20 — у 32 (tests/golden/retrieval.yaml): вдвое больший контекст
# добавляет один кейс.
SEARCH_PROVISION_LIMIT = 10
# Знаков заголовков и текста положений в контексте. Самый большой маршрут базы
# (оба договора «Альфа» и допсоглашение) — 13,4 тыс. знаков, самый большой
# контекст golden-набора — 16,8 тыс. (tests/golden/routing.yaml). Окно модели
# платформы неизвестно: лимит вдвое выше измеренного максимума и нужен, чтобы
# база с длинными договорами не увела контекст за окно молча.
CONTEXT_CHARACTER_LIMIT = 32_000

# Связи, по которым за документом идёт связанный. Смена редакции сюда не
# входит: какую редакцию брать, решает дата события, а не ссылка.
_FOLLOWED_LINKS = frozenset(
    {LinkKind.AMENDS, LinkKind.SUPPLEMENTS, LinkKind.REPLACES_PROVISION}
)


class RouteReason(Enum):
    """Почему документ попал в маршрут; значение идёт в журнал и интерфейс."""

    NAMED_BY_USER = "документ назван пользователем"
    COUNTERPARTY = "документ найден по контрагенту"
    LINKED = "связан с названным документом"
    EDITION_BY_EVENT_DATE = "редакция выбрана по дате события"
    SUGGESTED_BY_ASSISTANT = "подсказан ассистентом"


class Coverage(Enum):
    WHOLE = "whole"
    # Длинный документ: только положения, найденные в нём поиском.
    RELEVANT = "relevant"


@dataclass(frozen=True, slots=True)
class DocumentRoute:
    document: DocumentId
    coverage: Coverage
    # В порядке появления; у документа их бывает несколько: договор назван
    # пользователем и найден по контрагенту.
    reasons: tuple[RouteReason, ...]
    # Положения документа, которые остаются при любом сокращении: место, где
    # названа связь, изменённый пункт, его новая редакция и условие применения.
    required: tuple[ProvisionId, ...]


@dataclass(frozen=True, slots=True)
class EditionRoute:
    """Как для названного документа выбрана редакция."""

    family: EditionFamily
    # None — дата события названа неясно: в маршрут взяты все редакции. Исход
    # NoEditionOnDate — ни одна редакция в маршрут не взята, и ответ это говорит.
    choice: EditionChoice | None


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    # Сначала названное в диалоге, затем связанное по графу.
    routes: tuple[DocumentRoute, ...]
    editions: tuple[EditionRoute, ...]
    # В маршрут взяты документы всех кандидатов: ответ уточняет, о ком речь,
    # или отвечает по каждому.
    ambiguous: tuple[Mention, ...]
    # Названо то, чего в базе нет: ответ говорит об этом, маршрута нет.
    not_found: tuple[Mention, ...]
    # Своя компания, если её назвали: маршрута «все её документы» она не даёт.
    own_company: Counterparty | None


@dataclass(frozen=True, slots=True)
class ContextLimits:
    characters: int = CONTEXT_CHARACTER_LIMIT
    search_provisions: int = SEARCH_PROVISION_LIMIT


@dataclass(frozen=True, slots=True)
class ContextProvision:
    provision: AdmittedProvision
    # Пусто — положение пришло из общего поиска, а не по маршруту.
    reasons: tuple[RouteReason, ...]


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    # Маршрут в порядке плана и документа, затем поиск по убыванию оценки.
    provisions: tuple[ContextProvision, ...]
    # Документы маршрута, от которых из-за лимита осталась часть положений.
    reduced: tuple[DocumentId, ...]

    @property
    def admitted(self) -> tuple[AdmittedProvision, ...]:
        return tuple(item.provision for item in self.provisions)


def own_company(catalog: EntityCatalog) -> Counterparty | None:
    """Компания, чья это база: сторона большинства документов со сторонами.

    Она названа в шапке каждого своего договора, поэтому её упоминание ни один
    документ не выделяет. Признак — контрагент, у которого документов строго
    больше, чем у любого другого, и больше половины всех документов со
    сторонами. None — такого нет: в базе один договор, и стороны в нём равны.
    """
    ranked = sorted(catalog.counterparties, key=lambda item: -len(item.documents))
    if len(ranked) < 2 or len(ranked[0].documents) == len(ranked[1].documents):
        return None
    with_parties = {document for item in ranked for document in item.documents}
    leader = ranked[0]
    return leader if 2 * len(leader.documents) > len(with_parties) else None


def plan_retrieval(
    entities: DialogueEntities,
    event_date: EventDate | UnclearEventDate | None,
    today: date,
    catalog: EntityCatalog,
    graph: DocumentGraph,
    base: KnowledgeBase,
) -> RetrievalPlan:
    """Документы, которые диалог называет сам, и связанные с ними по графу."""
    own = own_company(catalog)
    resolutions = [mention.resolution for mention in entities.named_by_user]
    named = [entity for found in resolutions for entity in _candidates(found)]
    pointed: dict[DocumentId, list[RouteReason]] = {}
    for entity in named:
        reason = (
            RouteReason.COUNTERPARTY
            if isinstance(entity, Counterparty)
            else RouteReason.NAMED_BY_USER
        )
        _point(pointed, _documents_of(entity, own), reason)
    for entity in entities.suggested_by_assistant:
        _point(pointed, _documents_of(entity, own), RouteReason.SUGGESTED_BY_ASSISTANT)
    editions = tuple(_edition_routes(pointed, event_date, today, graph, base))
    routed = _with_chosen_editions(pointed, editions)
    for document in _linked(routed, graph):
        routed[document] = [RouteReason.LINKED]
    length = {
        document.id: sum(len(provision.text) for provision in document.provisions)
        for document in base.documents
    }
    required = _required(routed, graph, base)
    return RetrievalPlan(
        routes=tuple(
            DocumentRoute(
                document=document,
                coverage=(
                    Coverage.RELEVANT
                    if length[document] > LONG_DOCUMENT_CHARACTERS
                    else Coverage.WHOLE
                ),
                reasons=tuple(reasons),
                required=required.get(document, ()),
            )
            for document, reasons in routed.items()
        ),
        editions=editions,
        ambiguous=_mentions_of(entities, Ambiguous),
        not_found=_mentions_of(entities, NotFound),
        own_company=(
            own if own in (*named, *entities.suggested_by_assistant) else None
        ),
    )


def admitted_documents(
    base: KnowledgeBase, screening: Screening
) -> dict[DocumentId, tuple[AdmittedProvision, ...]]:
    """Допущенные положения по документам: из них исполнение берёт маршрут."""
    return {
        document.id: tuple(
            admitted
            for provision in document.provisions
            if (admitted := admit(document, provision, screening)) is not None
        )
        for document in base.documents
    }


def assemble_context(
    plan: RetrievalPlan,
    documents: Mapping[DocumentId, Sequence[AdmittedProvision]],
    ranking: Sequence[RankedProvision],
    limits: ContextLimits,
) -> RetrievedContext:
    """Положения маршрута и лучшие положения общего поиска без повторов.

    ranking — выдача общего поиска на всю глубину (у гибрида — поле hits): по
    ней же выбираются положения длинного документа. Маршрут важнее поиска: если
    он один не помещается в лимит, документы маршрута остаются все, но каждый
    сокращается по очереди, а из общего поиска не берётся ничего.
    """
    place = {hit.provision.id: index for index, hit in enumerate(ranking)}
    wanted = [
        _wanted(route, documents.get(route.document, ()), place, limits)
        for route in plan.routes
    ]
    fitting = _shared_out(wanted, limits.characters)
    routed: list[ContextProvision] = []
    reduced: list[DocumentId] = []
    for route, asked, kept in zip(plan.routes, wanted, fitting, strict=True):
        own = documents.get(route.document, ())
        chosen = {item.id for item in kept}
        routed.extend(
            ContextProvision(item, route.reasons) for item in own if item.id in chosen
        )
        if len(kept) < len(asked):
            reduced.append(route.document)
    taken = {item.provision.id for item in routed}
    room = limits.characters - sum(_size(item.provision) for item in routed)
    searched: list[ContextProvision] = []
    # Маршрут сокращён: место, отданное поиску, отнято бы у названного документа.
    best = () if reduced else ranking[: limits.search_provisions]
    for hit in best:
        if hit.provision.id in taken:
            continue
        if _size(hit.provision) > room:
            break
        room -= _size(hit.provision)
        searched.append(ContextProvision(hit.provision, ()))
    return RetrievedContext((*routed, *searched), tuple(reduced))


def _candidates(resolution: Resolved | Ambiguous | NotFound) -> tuple[Entity, ...]:
    if isinstance(resolution, Resolved):
        return (resolution.entity,)
    return resolution.candidates if isinstance(resolution, Ambiguous) else ()


def _documents_of(entity: Entity, own: Counterparty | None) -> tuple[DocumentId, ...]:
    if not isinstance(entity, Counterparty):
        return (entity.id,)
    return () if entity == own else entity.documents


def _point(
    pointed: dict[DocumentId, list[RouteReason]],
    documents: Iterable[DocumentId],
    reason: RouteReason,
) -> None:
    for document in documents:
        reasons = pointed.setdefault(document, [])
        if reason not in reasons:
            reasons.append(reason)


def _mentions_of(
    entities: DialogueEntities, outcome: type[Ambiguous] | type[NotFound]
) -> tuple[Mention, ...]:
    return tuple(
        mention
        for mention in entities.named_by_user
        if isinstance(mention.resolution, outcome)
    )


def _edition_routes(
    pointed: Mapping[DocumentId, Sequence[RouteReason]],
    event_date: EventDate | UnclearEventDate | None,
    today: date,
    graph: DocumentGraph,
    base: KnowledgeBase,
) -> Iterator[EditionRoute]:
    """Выбор редакции для каждого семейства, на документ которого указал диалог."""
    families = dict.fromkeys(
        family
        for document in pointed
        if (family := graph.family_of(document)) is not None
    )
    for family in families:
        if isinstance(event_date, UnclearEventDate):
            yield EditionRoute(family, None)
        elif event_date is None:
            yield EditionRoute(family, current_edition(family, today, base))
        else:
            yield EditionRoute(family, edition_on(family, event_date.period, base))


def _with_chosen_editions(
    pointed: Mapping[DocumentId, Sequence[RouteReason]],
    editions: Sequence[EditionRoute],
) -> dict[DocumentId, list[RouteReason]]:
    """Выбранные редакции встают на место названного документа своего семейства;
    причины, по которым он назван, переходят к ним."""
    choice_for = {
        edition.document: route
        for route in editions
        for edition in route.family.editions
    }
    routed: dict[DocumentId, list[RouteReason]] = {}
    for document, reasons in pointed.items():
        route = choice_for.get(document)
        if route is None:
            routed[document] = list(reasons)
            continue
        for reason in (*reasons, RouteReason.EDITION_BY_EVENT_DATE):
            _point(routed, _chosen_documents(route), reason)
    return routed


def _chosen_documents(route: EditionRoute) -> tuple[DocumentId, ...]:
    choice = route.choice
    if choice is None:
        return tuple(edition.document for edition in route.family.editions)
    if isinstance(choice, NoEditionOnDate):
        return ()
    if isinstance(choice, EditionsAcrossBoundary):
        return tuple(edition.document for edition in choice.editions)
    # Редакция действовала либо утратила силу неизвестно когда: во втором случае
    # её текст нужен ответу для оговорки.
    return (choice.edition.document,)


def _linked(routed: Iterable[DocumentId], graph: DocumentGraph) -> list[DocumentId]:
    """Допсоглашения, письма-дополнения и их основные документы, включая связи
    связанных: допсоглашение к договору, названному через другое допсоглашение."""
    named = list(routed)
    seen = list(named)
    for document in seen:
        for link in graph.links:
            if link.kind not in _FOLLOWED_LINKS or link.target is None:
                continue
            for near, far in ((link.source, link.target), (link.target, link.source)):
                if near == document and far not in seen:
                    seen.append(far)
    return seen[len(named) :]


def _required(
    routed: Mapping[DocumentId, Sequence[RouteReason]],
    graph: DocumentGraph,
    base: KnowledgeBase,
) -> dict[DocumentId, tuple[ProvisionId, ...]]:
    owner = {
        document.provision_id(provision): document.id
        for document in base.documents
        for provision in document.provisions
    }
    stated: list[ProvisionId | None] = []
    for link in graph.links:
        if link.kind in _FOLLOWED_LINKS and link.target in routed:
            condition = None if link.condition is None else link.condition.stated_in
            stated += [link.stated_in, link.target_provision, condition]
    for version in graph.provision_versions:
        if version.changed_by is not None:
            stated += [version.provision, version.stated_in]
    required: dict[DocumentId, list[ProvisionId]] = {}
    for provision in dict.fromkeys(stated):
        if provision is not None and owner[provision] in routed:
            required.setdefault(owner[provision], []).append(provision)
    return {document: tuple(found) for document, found in required.items()}


def _wanted(
    route: DocumentRoute,
    own: Sequence[AdmittedProvision],
    place: Mapping[ProvisionId, int],
    limits: ContextLimits,
) -> list[AdmittedProvision]:
    """Положения документа в порядке важности: обязательные, найденные поиском
    по месту в выдаче, остальные по порядку документа."""
    position = {item.id: index for index, item in enumerate(own)}
    ordered = sorted(
        own,
        key=lambda item: (
            item.id not in route.required,
            place.get(item.id, len(place)),
            position[item.id],
        ),
    )
    if route.coverage is Coverage.WHOLE:
        return ordered
    found = [
        item for item in ordered if item.id in place and item.id not in route.required
    ]
    relevant = {item.id for item in found[: limits.search_provisions]}
    return [
        item for item in ordered if item.id in route.required or item.id in relevant
    ]


def _shared_out(
    wanted: Sequence[Sequence[AdmittedProvision]], room: int
) -> list[list[AdmittedProvision]]:
    """Положения берутся по одному от каждого документа по кругу, пока есть место.

    Так при нехватке места в контексте остаются все документы маршрута — оба
    договора неоднозначного контрагента, договор и допсоглашение, — а не первый
    целиком. Документ, чьё очередное положение не поместилось, дальше не берётся.
    """
    kept: list[list[AdmittedProvision]] = [[] for _ in wanted]
    closed: set[int] = set()
    for layer in zip_longest(*wanted):
        for index, item in enumerate(layer):
            if item is None or index in closed:
                continue
            if _size(item) > room:
                closed.add(index)
                continue
            room -= _size(item)
            kept[index].append(item)
    return kept


def _size(provision: AdmittedProvision) -> int:
    return len(provision.heading) + len(provision.text)
