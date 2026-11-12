"""План маршрута и сборка контекста на выдуманной базе."""

from __future__ import annotations

from datetime import date

from lex_dates import PartialDate
from lex_dialogue import DialogueEntities
from lex_document_graph import build_graph
from lex_documents import DocumentId, ProvisionId
from lex_edition_choice import (
    EditionInForce,
    EditionsAcrossBoundary,
    MissingEdition,
    NoEditionOnDate,
)
from lex_entity_catalog import Counterparty, DocumentEntry, build_catalog
from lex_entity_recognition import (
    Ambiguous,
    Entity,
    Evidence,
    Mention,
    NotFound,
    Resolved,
)
from lex_event_date import EventDate, UnclearEventDate
from lex_knowledge_base import KnowledgeBase
from lex_retrieval import RankedProvision
from lex_routing import (
    LONG_DOCUMENT_CHARACTERS,
    ContextLimits,
    Coverage,
    RetrievalPlan,
    RetrievedContext,
    RouteReason,
    admitted_documents,
    assemble_context,
    own_company,
    plan_retrieval,
)
from lex_sanitizer import AdmittedProvision, Screening
from synthetic_links import (
    CONDITION,
    CONTRACT,
    NEW_RULES,
    NEW_TEXT,
    OLD_RULES,
    REPLACEMENT,
    agreement,
    knowledge_base,
    letter,
)

OTHER_CONTRACT = (
    CONTRACT.replace("С-21/2031", "К-07/2031")
    .replace("Гранит", "Кварц")
    .replace("6901000011", "6901000022")
)
LONG_CLAUSE = "Подрядчик ведёт журнал работ и передаёт его Заказчику по запросу. " * 3
PENALTY_CLAUSE = "Неустойка за просрочку работ — 0,1 % цены договора за каждый день."
LONG_CONTRACT = (
    CONTRACT.replace("С-21/2031", "Б-30/2031")
    .replace("Гранит", "Базальт")
    .replace("6901000011", "6901000033")
    + "".join(f"2.{number}. {LONG_CLAUSE}\n" for number in range(3, 60))
    + f"2.60. {PENALTY_CLAUSE}\n"
)
CONTRACT_ID = DocumentId("d01")
AGREEMENT_ID = DocumentId("d02")
OLD_RULES_ID = DocumentId("d03")
NEW_RULES_ID = DocumentId("d04")
LETTER_ID = DocumentId("d05")
OTHER_CONTRACT_ID = DocumentId("d06")
LONG_CONTRACT_ID = DocumentId("d07")
BASE = knowledge_base(
    CONTRACT,
    agreement(*REPLACEMENT),
    OLD_RULES,
    NEW_RULES,
    letter(),
    OTHER_CONTRACT,
    LONG_CONTRACT,
)
CATALOG = build_catalog(BASE)
GRAPH = build_graph(BASE)
DOCUMENTS = admitted_documents(BASE, Screening(()))
TODAY = date(2031, 9, 1)
DEFAULT_LIMITS = ContextLimits()
NOBODY = DialogueEntities(named_by_user=(), suggested_by_assistant=())


def _entry(document: DocumentId) -> DocumentEntry:
    return next(entry for entry in CATALOG.documents if entry.id == document)


def _party(name: str) -> Counterparty:
    return next(item for item in CATALOG.counterparties if item.name == name)


def _said(*entities: Entity) -> DialogueEntities:
    mentions = tuple(
        Mention("как в вопросе", Resolved(entity, Evidence.FULL_NAME))
        for entity in entities
    )
    return DialogueEntities(named_by_user=mentions, suggested_by_assistant=())


def _plan(
    entities: DialogueEntities,
    event_date: EventDate | UnclearEventDate | None = None,
    base: KnowledgeBase = BASE,
) -> RetrievalPlan:
    return plan_retrieval(entities, event_date, TODAY, CATALOG, GRAPH, base)


def _routes(plan: RetrievalPlan) -> dict[str, tuple[RouteReason, ...]]:
    return {route.document: route.reasons for route in plan.routes}


def _texts(provisions: tuple[ProvisionId, ...]) -> list[str]:
    found = [BASE.find(provision) for provision in provisions]
    return [item[1].text for item in found if item is not None]


def _ranking(*provisions: AdmittedProvision) -> tuple[RankedProvision, ...]:
    return tuple(
        RankedProvision(provision, float(len(provisions) - place))
        for place, provision in enumerate(provisions)
    )


def _context(
    plan: RetrievalPlan,
    ranking: tuple[RankedProvision, ...] = (),
    limits: ContextLimits = DEFAULT_LIMITS,
) -> RetrievedContext:
    return assemble_context(plan, DOCUMENTS, ranking, limits)


def _ids(context: RetrievedContext) -> list[str]:
    return [item.provision.id for item in context.provisions]


def test_named_document_goes_whole_with_its_amendment() -> None:
    plan = _plan(_said(_entry(CONTRACT_ID)))
    assert _routes(plan) == {
        CONTRACT_ID: (RouteReason.NAMED_BY_USER,),
        AGREEMENT_ID: (RouteReason.LINKED,),
    }
    assert {route.coverage for route in plan.routes} == {Coverage.WHOLE}
    assert plan.ambiguous == plan.not_found == ()


def test_changed_clause_its_new_text_and_condition_are_required() -> None:
    contract, amendment = _plan(_said(_entry(CONTRACT_ID))).routes
    assert len(contract.required) == 1
    assert "ул. Речная" in _texts(contract.required)[0]
    required = " ".join(_texts(amendment.required))
    assert NEW_TEXT in required
    assert CONDITION in required


def test_named_amendment_brings_its_contract() -> None:
    assert _routes(_plan(_said(_entry(AGREEMENT_ID)))) == {
        AGREEMENT_ID: (RouteReason.NAMED_BY_USER,),
        CONTRACT_ID: (RouteReason.LINKED,),
    }


def test_counterparty_gives_all_its_documents() -> None:
    assert _routes(_plan(_said(_party("ООО «Гранит»")))) == {
        CONTRACT_ID: (RouteReason.COUNTERPARTY,),
        AGREEMENT_ID: (RouteReason.COUNTERPARTY,),
    }


def test_document_named_twice_keeps_both_reasons() -> None:
    plan = _plan(_said(_party("ООО «Гранит»"), _entry(CONTRACT_ID)))
    assert _routes(plan)[CONTRACT_ID] == (
        RouteReason.COUNTERPARTY,
        RouteReason.NAMED_BY_USER,
    )


def test_ambiguous_counterparty_gives_documents_of_every_candidate() -> None:
    candidates = (_party("ООО «Гранит»"), _party("ООО «Кварц»"))
    mention = Mention("с тем подрядчиком", Ambiguous(candidates, Evidence.PARTIAL))
    plan = _plan(DialogueEntities((mention,), ()))
    assert list(_routes(plan)) == [CONTRACT_ID, AGREEMENT_ID, OTHER_CONTRACT_ID]
    assert plan.ambiguous == (mention,)


def test_own_company_is_the_party_of_most_documents() -> None:
    own = own_company(CATALOG)
    assert own is not None
    assert own.name == "ООО «Бирюза»"


def test_single_contract_has_no_own_company() -> None:
    assert own_company(build_catalog(knowledge_base(CONTRACT))) is None


def test_two_contracts_with_one_counterparty_have_no_own_company() -> None:
    # У обеих сторон по два документа: кто из них «своя», по базе не понять.
    catalog = build_catalog(knowledge_base(CONTRACT, agreement(*REPLACEMENT)))
    assert own_company(catalog) is None


def test_own_company_gives_no_route() -> None:
    plan = _plan(_said(_party("ООО «Бирюза»")))
    assert plan.routes == ()
    assert plan.own_company == _party("ООО «Бирюза»")


def test_own_company_next_to_a_counterparty_does_not_widen_the_route() -> None:
    plan = _plan(_said(_party("ООО «Бирюза»"), _party("ООО «Кварц»")))
    assert list(_routes(plan)) == [OTHER_CONTRACT_ID]


def test_assistant_suggestion_is_a_reason_of_its_own() -> None:
    suggested = (_party("ООО «Кварц»"), _entry(CONTRACT_ID))
    plan = _plan(DialogueEntities((), suggested))
    assert _routes(plan) == {
        OTHER_CONTRACT_ID: (RouteReason.SUGGESTED_BY_ASSISTANT,),
        CONTRACT_ID: (RouteReason.SUGGESTED_BY_ASSISTANT,),
        AGREEMENT_ID: (RouteReason.LINKED,),
    }
    assert plan.own_company is None


def test_own_company_suggested_by_assistant_gives_no_route() -> None:
    plan = _plan(DialogueEntities((), (_party("ООО «Бирюза»"),)))
    assert plan.routes == ()
    assert plan.own_company is not None


def test_event_date_chooses_the_edition_instead_of_the_named_one() -> None:
    asked = EventDate(PartialDate.of_year(2030), "в 2030 году")
    plan = _plan(_said(_entry(NEW_RULES_ID)), asked)
    # Письмо дополняет новую редакцию: к старой оно не относится.
    assert _routes(plan) == {
        OLD_RULES_ID: (RouteReason.NAMED_BY_USER, RouteReason.EDITION_BY_EVENT_DATE)
    }
    (edition,) = plan.editions
    assert isinstance(edition.choice, EditionInForce)
    assert edition.choice.date_stated


def test_without_event_date_the_current_edition_is_taken() -> None:
    plan = _plan(_said(_entry(OLD_RULES_ID)))
    assert _routes(plan) == {
        NEW_RULES_ID: (RouteReason.NAMED_BY_USER, RouteReason.EDITION_BY_EVENT_DATE),
        LETTER_ID: (RouteReason.LINKED,),
    }
    (edition,) = plan.editions
    assert isinstance(edition.choice, EditionInForce)
    assert not edition.choice.date_stated


def test_period_across_the_change_takes_both_editions() -> None:
    asked = EventDate(PartialDate.of_year(2031), "в 2031 году")
    plan = _plan(_said(_entry(NEW_RULES_ID)), asked)
    assert list(_routes(plan)) == [OLD_RULES_ID, NEW_RULES_ID, LETTER_ID]
    assert isinstance(plan.editions[0].choice, EditionsAcrossBoundary)


def test_unclear_event_date_takes_every_edition() -> None:
    unclear = UnclearEventDate(candidates=(), incomplete=("в августе",))
    plan = _plan(_said(_entry(NEW_RULES_ID)), unclear)
    assert list(_routes(plan)) == [OLD_RULES_ID, NEW_RULES_ID, LETTER_ID]
    assert plan.editions[0].choice is None


def test_date_before_the_first_edition_leaves_the_plan_with_the_reason_only() -> None:
    asked = EventDate(PartialDate.of_year(2020), "в 2020 году")
    plan = _plan(_said(_entry(NEW_RULES_ID)), asked)
    assert plan.routes == ()
    assert plan.editions[0].choice == NoEditionOnDate(
        MissingEdition.BEFORE_FIRST_EDITION
    )


def test_named_document_missing_from_the_base_is_kept_in_the_plan() -> None:
    mention = Mention("С-21/2019", NotFound(similar=(_entry(CONTRACT_ID),)))
    plan = _plan(DialogueEntities((mention,), ()))
    assert plan.routes == ()
    assert plan.not_found == (mention,)


def test_question_without_entities_is_answered_by_search_alone() -> None:
    plan = _plan(NOBODY)
    assert plan == RetrievalPlan((), (), (), (), None)
    first, second, *_ = DOCUMENTS[OTHER_CONTRACT_ID]
    context = _context(plan, _ranking(second, first))
    assert _ids(context) == [second.id, first.id]
    assert all(item.reasons == () for item in context.provisions)


def test_long_document_is_routed_by_relevant_provisions() -> None:
    length = sum(len(item.text) for item in DOCUMENTS[LONG_CONTRACT_ID])
    assert length > LONG_DOCUMENT_CHARACTERS
    plan = _plan(_said(_entry(LONG_CONTRACT_ID)))
    (route,) = plan.routes
    assert route.coverage is Coverage.RELEVANT
    own = DOCUMENTS[LONG_CONTRACT_ID]
    penalty, subject = own[-1], own[1]
    foreign = DOCUMENTS[CONTRACT_ID][0]
    context = _context(plan, _ranking(penalty, foreign, subject))
    # Из длинного документа — только найденное, в порядке документа.
    assert _ids(context) == [subject.id, penalty.id, foreign.id]
    assert context.provisions[0].reasons == (RouteReason.NAMED_BY_USER,)
    assert context.reduced == ()


def test_long_document_gives_no_more_provisions_than_the_search_limit() -> None:
    plan = _plan(_said(_entry(LONG_CONTRACT_ID)))
    own = DOCUMENTS[LONG_CONTRACT_ID]
    context = _context(plan, _ranking(*own), ContextLimits(search_provisions=4))
    assert _ids(context) == [item.id for item in own[:4]]


def test_route_comes_first_in_document_order_then_search_without_repeats() -> None:
    plan = _plan(_said(_entry(CONTRACT_ID)))
    contract, amendment = DOCUMENTS[CONTRACT_ID], DOCUMENTS[AGREEMENT_ID]
    other = DOCUMENTS[OTHER_CONTRACT_ID]
    context = _context(plan, _ranking(other[2], contract[3], other[1], amendment[0]))
    assert _ids(context) == [
        *(item.id for item in contract),
        *(item.id for item in amendment),
        other[2].id,
        other[1].id,
    ]
    assert [item.reasons for item in context.provisions[-2:]] == [(), ()]
    assert context.provisions[0].reasons == (RouteReason.NAMED_BY_USER,)
    assert context.provisions[len(contract)].reasons == (RouteReason.LINKED,)


def test_search_adds_only_its_best_provisions() -> None:
    other = DOCUMENTS[OTHER_CONTRACT_ID]
    context = _context(
        _plan(NOBODY), _ranking(*other), ContextLimits(search_provisions=2)
    )
    assert _ids(context) == [other[0].id, other[1].id]


def test_search_stops_at_the_character_limit() -> None:
    other = DOCUMENTS[OTHER_CONTRACT_ID]
    room = len(other[0].heading) + len(other[0].text)
    context = _context(_plan(NOBODY), _ranking(*other), ContextLimits(characters=room))
    assert _ids(context) == [other[0].id]


def test_route_over_the_limit_keeps_every_document_and_the_required() -> None:
    plan = _plan(_said(_entry(CONTRACT_ID)))
    contract, amendment = plan.routes
    whole = _context(plan)
    size = sum(len(item.heading) + len(item.text) for item in whole.admitted)
    foreign = DOCUMENTS[OTHER_CONTRACT_ID][0]
    limits = ContextLimits(characters=size // 2)
    context = _context(plan, _ranking(foreign), limits)
    kept = set(_ids(context))
    assert context.reduced == (CONTRACT_ID, AGREEMENT_ID)
    assert set(contract.required) | set(amendment.required) <= kept
    # Маршрут сокращён — общий поиск места не получает.
    assert foreign.id not in kept
    assert sum(len(item.heading) + len(item.text) for item in context.admitted) <= (
        limits.characters
    )
    assert _ids(context) == [item for item in _ids(whole) if item in kept]


def test_reduced_route_prefers_provisions_found_by_search() -> None:
    plan = _plan(_said(_entry(OTHER_CONTRACT_ID)))
    own = DOCUMENTS[OTHER_CONTRACT_ID]
    last = own[-1]
    limits = ContextLimits(characters=len(last.heading) + len(last.text))
    context = _context(plan, _ranking(last), limits)
    assert _ids(context) == [last.id]
    assert context.reduced == (OTHER_CONTRACT_ID,)
