"""Ответ с проверкой: генерация, одна перегенерация по отчёту и безопасный откат.

Верификатор детерминирован, поэтому его отчёт — надёжная обратная связь для
второй попытки. Попытка одна: дополнительный вызов модели на вопрос общий
с повтором по контракту. Что не подтвердилось и после неё — удаляется.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from lex_answer_contract import (
    NO_DATA_ANSWER,
    SCENARIOS_ANSWERED_BY_TEMPLATE,
    Generation,
    GroundedAnswer,
    ProblemKind,
    UnconfirmedAnswer,
    ask,
    generate,
)
from lex_document_graph import DocumentGraph
from lex_documents import ProvisionId
from lex_fact_extraction import DateFact, Fact, Unit, find_facts
from lex_models import ChatTurn, JsonChatModel
from lex_prompt_boundary import Boundary
from lex_sanitizer import AdmittedProvision, Screening
from lex_verification import VerificationReport, apply_report, verify

# Перечень дописывается после повтора правил, как ANSWER_CORRECTION_V1. В нём
# только наш текст и идентификаторы положений контекста: прошлый ответ модели
# и текст из данных в запрос не возвращаются, поэтому утверждение адресуется
# положениями из его `ids` и видом величины, а не номером или цитатой.
REGENERATION_REQUEST_V1 = """\
Предыдущий ответ не прошёл сверку с текстом положений. Отклонено:
{violations}
Ответь заново на тот же вопрос: один JSON-объект по примеру из правил. \
Величину, её единицу и слова «не более», «не менее», «свыше», «до» бери из текста \
положения, которое называешь в ids. Если нужной величины в положениях нет, \
утверждение о ней не пиши. Остальную часть ответа не сокращай."""
_VIOLATION_LINE = "- claims.text при ids {ids}: {measure} — {violation}"

_DATE_MEASURE = "дата"
_MEASURES: Mapping[Unit, str] = {
    Unit.RUBLE: "сумма в рублях",
    Unit.PERCENT: "процент",
    Unit.CALENDAR_DAY: "срок в календарных днях",
    Unit.WORKING_DAY: "срок в рабочих днях",
    Unit.BANKING_DAY: "срок в банковских днях",
    Unit.DAY: "срок в днях",
    Unit.WEEK: "срок в неделях",
    Unit.MONTH: "срок в месяцах",
    Unit.YEAR: "срок в годах",
    Unit.HOUR: "срок в часах",
    Unit.DAY_OF_MONTH: "день месяца",
    Unit.NONE: "число без единицы",
}
# Модель не дала текста, который можно разобрать: проверять было нечего.
_NO_REPLY = frozenset(
    {
        ProblemKind.NO_REPLY,
        ProblemKind.CONTENT_FILTERED,
        ProblemKind.EMPTY,
        ProblemKind.TRUNCATED,
    }
)


class AnswerPath(Enum):
    """Каким путём получен итоговый ответ. Значение — метка для журнала."""

    # Верификатор ничего не снял.
    ACCEPTED = "accepted"
    # Взят ответ второй попытки: подтверждённых утверждений в нём больше.
    REGENERATED = "regenerated"
    # Взят первый ответ без неподтверждённых утверждений и строк `missing`.
    CLEANED = "cleaned"
    # Откат: положения по вопросу есть, но утверждения о них сняты все.
    # Частичный ответ с указанием документа, не отказ.
    UNCONFIRMED = "unconfirmed"
    # Откат: модель ответила, но ни одного утверждения с опорой не назвала.
    NO_DATA = "no_data"
    # Откат: модель не ответила — сбой прокси, фильтр, пустой или оборванный ответ.
    MODEL_FAILED = "model_failed"


@dataclass(frozen=True, slots=True)
class Attempt:
    """Ответ модели, прошедший контракт, и отчёт его проверки."""

    generation: Generation
    report: VerificationReport


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    answer: GroundedAnswer | UnconfirmedAnswer
    path: AnswerPath
    first: Attempt
    # None — перегенерация не запускалась.
    second: Attempt | None

    @property
    def model_calls(self) -> int:
        """Не больше двух: повтор по контракту и перегенерация исключают друг друга."""
        return 1 + self.first.generation.extra_call_spent + (self.second is not None)


def generate_verified(
    model: JsonChatModel,
    turns: Sequence[ChatTurn],
    context: Sequence[AdmittedProvision],
    boundary: Boundary,
    question: str,
    screening: Screening,
    graph: DocumentGraph,
) -> VerifiedAnswer:
    """Ответ, в котором нет ничего, что отклонил верификатор.

    Из двух ответов берётся тот, где подтверждённых утверждений больше; при
    равенстве — первый: вторая попытка не должна ничего ухудшать. `turns` —
    запрос из `assemble_prompt`, `context` — показанные модели положения.
    """
    first = _checked(
        generate(model, turns, context, boundary), context, question, screening, graph
    )
    cleaned = apply_report(first.generation, first.report)
    if not cleaned.regeneration_worthwhile:
        path = AnswerPath.ACCEPTED if first.report.passed else AnswerPath.CLEANED
        return _concluded(cleaned.answer, path, first, None)
    request = regeneration_request(turns, first.generation.answer, first.report)
    reading = ask(model, request, context, boundary)
    second = _checked(
        Generation(reading.grounded or NO_DATA_ANSWER, True, reading.problems),
        context,
        question,
        screening,
        graph,
    )
    recleaned = apply_report(second.generation, second.report).answer
    if len(recleaned.claims) > len(cleaned.answer.claims):
        return _concluded(recleaned, AnswerPath.REGENERATED, first, second)
    return _concluded(cleaned.answer, AnswerPath.CLEANED, first, second)


def regeneration_request(
    turns: Sequence[ChatTurn], answer: GroundedAnswer, report: VerificationReport
) -> list[ChatTurn]:
    """Тот же запрос, к последнему сообщению которого дописано, что отклонено.

    `report` составлен по `answer`. В перечень идут утверждения, которые вторая
    попытка может исправить; строки `missing` и повтор карантина — нет.
    """
    lines = [
        _VIOLATION_LINE.format(
            ids=", ".join(claim.provisions),
            measure=_measure(fact),
            violation=fact_check.violation.value,
        )
        for claim, check in zip(answer.claims, report.claims, strict=True)
        if check.repairable
        for fact, fact_check in zip(find_facts(claim.text), check.facts, strict=True)
        if fact_check.violation is not None
    ]
    *before, last = turns
    addition = REGENERATION_REQUEST_V1.format(
        violations="\n".join(dict.fromkeys(lines))
    )
    return [*before, ChatTurn(last.role, f"{last.content}\n{addition}")]


def _checked(
    generation: Generation,
    context: Sequence[AdmittedProvision],
    question: str,
    screening: Screening,
    graph: DocumentGraph,
) -> Attempt:
    report = verify(generation.answer, context, question, screening, graph)
    return Attempt(generation, report)


def _concluded(
    answer: GroundedAnswer, path: AnswerPath, first: Attempt, second: Attempt | None
) -> VerifiedAnswer:
    stated = answer.claims or answer.missing
    if stated or answer.scenario in SCENARIOS_ANSWERED_BY_TEMPLATE:
        return VerifiedAnswer(answer, path, first, second)
    attempts = (first,) if second is None else (first, second)
    rejected = _rejected_provisions(attempts)
    if rejected:
        return VerifiedAnswer(
            UnconfirmedAnswer(rejected), AnswerPath.UNCONFIRMED, first, second
        )
    silent = any(problem.kind in _NO_REPLY for problem in first.generation.problems)
    fallback = AnswerPath.MODEL_FAILED if silent else AnswerPath.NO_DATA
    return VerifiedAnswer(NO_DATA_ANSWER, fallback, first, second)


def _rejected_provisions(attempts: Sequence[Attempt]) -> tuple[ProvisionId, ...]:
    """Положения снятых утверждений: белый список и сверку цитаты они прошли."""
    cited = (
        provision
        for attempt in attempts
        for check in attempt.report.claims
        if not check.confirmed
        for provision in check.provisions
    )
    return tuple(dict.fromkeys(cited))


def _measure(fact: Fact) -> str:
    return _DATE_MEASURE if isinstance(fact, DateFact) else _MEASURES[fact.unit]
