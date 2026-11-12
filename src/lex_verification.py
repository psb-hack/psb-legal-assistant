"""Проверка покрытия величин: каждое число утверждения есть в его положениях.

Сверяются значения, единицы, операторы и даты — без модели. Смысл утверждения
(«450 000 — это штраф или арендная плата») этот слой не проверяет.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict

from lex_answer_contract import (
    NO_DATA_ANSWER,
    AnswerVerdict,
    Generation,
    GroundedAnswer,
    GroundedClaim,
)
from lex_document_graph import DocumentGraph
from lex_document_numbers import DocumentNumber, find_document_numbers
from lex_documents import ProvisionId
from lex_errors import VerificationError
from lex_fact_extraction import Bound, DateFact, Fact, Unit, find_facts
from lex_normalization import comparison_key
from lex_sanitizer import AdmittedProvision, Screening
from lex_scenarios import Scenario

# Столько слов подряд из карантинного фрагмента в утверждении — уже пересказ.
QUARANTINE_MATCH_WORDS = 5

_WORD = re.compile(r"[^\W_]+")
# «…трёх месяцев. Срок…», перевод строки: дальше оператор не действует.
_SENTENCE_END = re.compile(r"[.;!?]\s+[А-ЯЁA-Z]|\n")


class ViolationKind(Enum):
    """Чем утверждение не подтверждено. Значение — строка для интерфейса."""

    VALUE_NOT_FOUND = "величины нет ни в одном положении контекста"
    FOREIGN_SOURCE = (
        "величина есть в контексте, но не в названных положениях и не в их изменениях"
    )
    UNIT_MISMATCH = "в названном положении у этого числа другая единица"
    BOUND_MISMATCH = (
        "в названном положении у величины другой оператор: «не более», «свыше», «до»"
    )
    VAT_MISMATCH = "в названном положении у суммы другой признак НДС"
    DATE_MORE_PRECISE = "дата названа точнее, чем в названном положении"
    QUARANTINED_VALUE = "величина есть только во фрагменте базы из карантина"
    QUARANTINED_WORDING = "текст повторяет фрагмент базы из карантина"
    MISSING_STATES_VALUE = "в перечне недостающего названа величина не из вопроса"
    MISSING_NAMES_DOCUMENT = "в перечне недостающего назван документ из контекста"


_QUARANTINE_VIOLATIONS = frozenset(
    {ViolationKind.QUARANTINED_VALUE, ViolationKind.QUARANTINED_WORDING}
)


class FactSupport(Enum):
    CONFIRMED = "confirmed"
    # Повтор слов пользователя: не провал, но и не подтверждённый факт.
    FROM_QUESTION = "from_question"
    REJECTED = "rejected"


class FactCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Величина словами утверждения; None — утверждение повторяет карантин.
    wording: str | None
    support: FactSupport
    confirmed_by: ProvisionId | None
    violation: ViolationKind | None


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # None — текст повторяет карантин: в отчёт он не попадает, чтобы не уйти
    # ни в интерфейс, ни в повторный запрос к модели.
    text: str | None
    provisions: tuple[ProvisionId, ...]
    confirmed: bool
    facts: tuple[FactCheck, ...]
    violations: tuple[ViolationKind, ...]

    @property
    def repairable(self) -> bool:
        """Отклонено расхождением с названным положением, а не карантином.

        Такое утверждение вторая попытка может исправить. Повтор карантинного
        текста она не исправит: его просто удаляют.
        """
        return not self.confirmed and not _QUARANTINE_VIOLATIONS.intersection(
            self.violations
        )


class MissingCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    violations: tuple[ViolationKind, ...]


class VerificationReport(BaseModel):
    """Результат по каждому утверждению и каждой строке `missing`, по порядку."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: tuple[ClaimCheck, ...]
    missing: tuple[MissingCheck, ...]

    @property
    def passed(self) -> bool:
        return all(claim.confirmed for claim in self.claims) and not any(
            item.violations for item in self.missing
        )


@dataclass(frozen=True, slots=True)
class Correction:
    # Ответ без неподтверждённых утверждений и строк `missing`.
    answer: GroundedAnswer
    # Есть утверждение, которое вторая попытка может исправить, а дополнительный
    # вызов модели ещё не израсходован.
    regeneration_worthwhile: bool


@dataclass(frozen=True, slots=True)
class _SourceFact:
    fact: Fact
    # Оператор самой величины и унаследованный от предыдущей в том же
    # предложении: «не может превышать трёх месяцев, а для … — шести месяцев».
    bounds: frozenset[Bound]


@dataclass(frozen=True, slots=True)
class _Evidence:
    by_provision: Mapping[ProvisionId, tuple[_SourceFact, ...]]
    question: tuple[Fact, ...]
    quarantined: tuple[_SourceFact, ...]
    # Цепочки слов, которые есть в карантине и которых нет в контексте.
    quarantined_wording: frozenset[tuple[str, ...]]
    document_numbers: frozenset[DocumentNumber]

    @property
    def context(self) -> tuple[_SourceFact, ...]:
        return tuple(fact for facts in self.by_provision.values() for fact in facts)


# От дальнего расхождения к ближнему: в отчёт идёт самое близкое к подтверждению.
_CLOSENESS = (
    ViolationKind.VALUE_NOT_FOUND,
    ViolationKind.DATE_MORE_PRECISE,
    ViolationKind.UNIT_MISMATCH,
    ViolationKind.BOUND_MISMATCH,
    ViolationKind.VAT_MISMATCH,
)
# Источник с любым из этих операторов называет границу, но не подтверждает
# слово «порог».
_NOT_A_THRESHOLD = frozenset({Bound.EXACT, Bound.RANGE})


def verify(
    answer: GroundedAnswer,
    context: Sequence[AdmittedProvision],
    question: str,
    screening: Screening,
    graph: DocumentGraph,
) -> VerificationReport:
    """Сверяет величины ответа с положениями контекста, вопросом и карантином.

    `context` — положения, показанные модели; `question` — слова пользователя.
    """
    evidence = _evidence(context, question, screening)
    return VerificationReport(
        claims=tuple(_check_claim(claim, evidence, graph) for claim in answer.claims),
        missing=tuple(
            _check_missing(item, answer.scenario, evidence) for item in answer.missing
        ),
    )


def apply_report(generation: Generation, report: VerificationReport) -> Correction:
    """Убирает из ответа всё, что отчёт не подтвердил.

    Перегенерацию эта функция не делает: она только сообщает, есть ли в ней
    смысл. Повтор запроса в `generate` и перегенерация делят один вызов.
    """
    answer = generation.answer
    if (len(report.claims), len(report.missing)) != (
        len(answer.claims),
        len(answer.missing),
    ):
        raise VerificationError("отчёт составлен по другому ответу")
    claims = tuple(
        claim
        for claim, check in zip(answer.claims, report.claims, strict=True)
        if check.confirmed
    )
    missing = tuple(
        item
        for item, check in zip(answer.missing, report.missing, strict=True)
        if not check.violations
    )
    if report.passed:
        return Correction(answer, False)
    # Строку `missing` и повтор карантина дешевле удалить, чем перегенерировать.
    worthwhile = not generation.extra_call_spent and any(
        check.repairable for check in report.claims
    )
    if not claims and not missing:
        return Correction(NO_DATA_ANSWER, worthwhile)
    # «Да» и «нет» опираются на все утверждения ответа, как и в контракте.
    intact = len(claims) == len(answer.claims)
    verdict = answer.verdict if intact else AnswerVerdict.NONE
    cleaned = GroundedAnswer(answer.scenario, claims, missing, verdict)
    return Correction(cleaned, worthwhile)


def _evidence(
    context: Sequence[AdmittedProvision], question: str, screening: Screening
) -> _Evidence:
    quarantined = [decision.original for decision in screening.quarantined]
    shown = {
        chain
        for provision in context
        for chain in _word_chains(f"{provision.heading}\n{provision.text}")
    }
    return _Evidence(
        by_provision={
            # Заголовок несёт реквизиты документа: «в редакции от 10.01.2023».
            provision.id: (
                *_source_facts(provision.heading),
                *_source_facts(provision.text),
            )
            for provision in context
        },
        question=find_facts(question),
        quarantined=tuple(fact for text in quarantined for fact in _source_facts(text)),
        quarantined_wording=frozenset(
            chain for text in quarantined for chain in _word_chains(text)
        )
        - shown,
        document_numbers=frozenset(
            mention.number
            for provision in context
            for mention in find_document_numbers(provision.heading)
        ),
    )


def _source_facts(text: str) -> tuple[_SourceFact, ...]:
    found: list[_SourceFact] = []
    for fact in find_facts(text):
        bounds = {fact.bound}
        previous = found[-1] if found else None
        if (
            previous is not None
            and fact.bound is Bound.EXACT
            and _same_measure(previous.fact, fact)
            and not _SENTENCE_END.search(text[previous.fact.end : fact.start])
        ):
            bounds |= previous.bounds
        found.append(_SourceFact(fact, frozenset(bounds)))
    return tuple(found)


def _same_measure(first: Fact, second: Fact) -> bool:
    if isinstance(first, DateFact) or isinstance(second, DateFact):
        return isinstance(first, DateFact) and isinstance(second, DateFact)
    return first.kind is second.kind


def _word_chains(text: str) -> set[tuple[str, ...]]:
    words = _WORD.findall(comparison_key(text))
    return {
        tuple(words[start : start + QUARANTINE_MATCH_WORDS])
        for start in range(len(words) - QUARANTINE_MATCH_WORDS + 1)
    }


def _check_claim(
    claim: GroundedClaim, evidence: _Evidence, graph: DocumentGraph
) -> ClaimCheck:
    sources = _with_amendments(claim.provisions, evidence, graph)
    facts = [
        _check_fact(claim.text, fact, sources, evidence)
        for fact in find_facts(claim.text)
    ]
    violations = [check.violation for check in facts if check.violation is not None]
    if _word_chains(claim.text) & evidence.quarantined_wording:
        violations.append(ViolationKind.QUARANTINED_WORDING)
    if _QUARANTINE_VIOLATIONS.intersection(violations):
        facts = [check.model_copy(update={"wording": None}) for check in facts]
    return ClaimCheck(
        text=None if _QUARANTINE_VIOLATIONS.intersection(violations) else claim.text,
        provisions=claim.provisions,
        confirmed=not violations,
        facts=tuple(facts),
        violations=tuple(dict.fromkeys(violations)),
    )


def _with_amendments(
    cited: Sequence[ProvisionId], evidence: _Evidence, graph: DocumentGraph
) -> tuple[ProvisionId, ...]:
    """Названные положения и положения контекста, которые меняют их по графу.

    Происхождение: величину пункта подтверждает сам пункт или его новая
    редакция из допсоглашения, но не посторонний документ.
    """
    amending = (
        version.stated_in
        for provision in cited
        for version in graph.versions_of(provision)
    )
    return tuple(
        provision
        for provision in dict.fromkeys((*cited, *amending))
        if provision in evidence.by_provision
    )


def _check_fact(
    text: str, fact: Fact, sources: Sequence[ProvisionId], evidence: _Evidence
) -> FactCheck:
    wording = text[fact.start : fact.end]
    for provision in sources:
        if _violation(fact, evidence.by_provision[provision]) is None:
            return FactCheck(
                wording=wording,
                support=FactSupport.CONFIRMED,
                confirmed_by=provision,
                violation=None,
            )
    if any(_repeats(fact, asked) for asked in evidence.question):
        return FactCheck(
            wording=wording,
            support=FactSupport.FROM_QUESTION,
            confirmed_by=None,
            violation=None,
        )
    cited = (found for one in sources for found in evidence.by_provision[one])
    violation = _violation(fact, cited)
    in_context = _violation(fact, evidence.context) is None
    if not in_context and _violation(fact, evidence.quarantined) is None:
        violation = ViolationKind.QUARANTINED_VALUE
    elif in_context and violation is ViolationKind.VALUE_NOT_FOUND:
        violation = ViolationKind.FOREIGN_SOURCE
    return FactCheck(
        wording=wording,
        support=FactSupport.REJECTED,
        confirmed_by=None,
        violation=violation,
    )


def _violation(claimed: Fact, sources: Iterable[_SourceFact]) -> ViolationKind | None:
    """None — величина подтверждена; иначе самое близкое расхождение."""
    closest = ViolationKind.VALUE_NOT_FOUND
    for source in sources:
        difference = _difference(claimed, source)
        if difference is None:
            return None
        closest = max(closest, difference, key=_CLOSENESS.index)
    return closest


def _difference(claimed: Fact, source: _SourceFact) -> ViolationKind | None:
    known = source.fact
    if isinstance(claimed, DateFact) or isinstance(known, DateFact):
        if isinstance(claimed, DateFact) and isinstance(known, DateFact):
            return _date_difference(claimed, known, source.bounds)
        return ViolationKind.VALUE_NOT_FOUND
    if (claimed.value, claimed.upper) != (known.value, known.upper):
        return ViolationKind.VALUE_NOT_FOUND
    # Число без единицы единицу не утверждает: его подтверждает любая.
    if claimed.unit not in (Unit.NONE, known.unit):
        return ViolationKind.UNIT_MISMATCH
    if not _bound_confirmed(claimed.bound, source.bounds):
        return ViolationKind.BOUND_MISMATCH
    # НДС сверяется, только если ответ его назвал.
    if claimed.vat not in (None, known.vat):
        return ViolationKind.VAT_MISMATCH
    return None


def _date_difference(
    claimed: DateFact, known: DateFact, bounds: frozenset[Bound]
) -> ViolationKind | None:
    if _within(known, claimed):
        # Ответ не точнее источника: «в 2024 году» при «1 августа 2024 г.».
        return None if claimed.bound in bounds else ViolationKind.BOUND_MISMATCH
    if _within(claimed, known):
        return ViolationKind.DATE_MORE_PRECISE
    return ViolationKind.VALUE_NOT_FOUND


def _within(inner: DateFact, outer: DateFact) -> bool:
    return (
        outer.period.start <= inner.period.start
        and inner.period.end <= outer.period.end
    )


def _bound_confirmed(claimed: Bound, bounds: frozenset[Bound]) -> bool:
    if claimed is Bound.THRESHOLD:
        # «Порог — 15 000 000» подтверждает «свыше 15 000 000», «не более …».
        return bool(bounds - _NOT_A_THRESHOLD)
    return claimed in bounds


def _repeats(claimed: Fact, asked: Fact) -> bool:
    """Величина утверждения повторяет названную пользователем.

    Оператор не сверяется: «а не 500 000» отвечает на «почему 500 000 рублей».
    """
    if isinstance(claimed, DateFact) or isinstance(asked, DateFact):
        return (
            isinstance(claimed, DateFact)
            and isinstance(asked, DateFact)
            and _within(asked, claimed)
        )
    same_value = (claimed.value, claimed.upper) == (asked.value, asked.upper)
    return same_value and (
        Unit.NONE in (claimed.unit, asked.unit) or claimed.unit is asked.unit
    )


def _check_missing(item: str, scenario: Scenario, evidence: _Evidence) -> MissingCheck:
    """Свободный текст без опоры на положение: величин и документов в нём нет.

    В сценарии `ambiguous` строка называет, что уточнить, и может перечислять
    варианты из контекста: там допустимы его величины и документы.
    """
    clarifies = scenario is Scenario.AMBIGUOUS
    violations: list[ViolationKind] = []
    for fact in find_facts(item):
        asked = any(_repeats(fact, one) for one in evidence.question)
        shown = clarifies and _violation(fact, evidence.context) is None
        if not asked and not shown:
            violations.append(ViolationKind.MISSING_STATES_VALUE)
    named = {mention.number for mention in find_document_numbers(item)}
    if not clarifies and named & evidence.document_numbers:
        violations.append(ViolationKind.MISSING_NAMES_DOCUMENT)
    return MissingCheck(text=item, violations=tuple(dict.fromkeys(violations)))
