"""Контракт ответа модели: схема JSON, сверка с контекстом и один повтор.

Модель возвращает JSON по схеме ModelAnswer. Поля идут в порядке генерации:
сначала опора (идентификаторы положений и цитата), затем формулировка, вердикт —
последним. Текст ссылок и прямой ответ первой фразой собирает рендерер.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, ValidationError

from lex_documents import ProvisionId
from lex_errors import ModelGatewayError
from lex_models import ChatTurn, FinishReason, JsonChatModel, JsonCompletion
from lex_normalization import comparison_key
from lex_prompt_boundary import Boundary, mentions_boundary
from lex_sanitizer import AdmittedProvision
from lex_scenarios import Scenario

# Короче этого цитата ничего не доказывает: предлог или одно число найдётся
# в любом положении. Длина считается по ключу сравнения.
MIN_QUOTE_LENGTH = 10

# Перечень ошибок дописывается к запросу после повтора правил. Прошлый ответ
# модели в запрос не возвращается: в нём может быть текст, которому модель
# последовала из данных, а реплика ассистента весит больше блока данных.
ANSWER_CORRECTION_V1 = """\
Предыдущий ответ отклонён проверкой. Причины:
{problems}
Ответь заново: один JSON-объект по примеру из правил."""
_NO_PROVISIONS_IN_REQUEST = "в запросе нет блоков provision"


class AnswerVerdict(Enum):
    """Прямой ответ на вопрос «да или нет»; у остальных вопросов вердикта нет."""

    YES = "yes"
    NO = "no"
    NONE = "none"


class ModelClaim(BaseModel):
    """Одно утверждение: положения, дословная цитата из них, затем формулировка."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    ids: tuple[str, ...]
    quote: str
    text: str


class ModelAnswer(BaseModel):
    """То, что возвращает модель. Порядок полей — порядок генерации."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scenario: Scenario
    claims: tuple[ModelClaim, ...]
    missing: tuple[str, ...]
    verdict: AnswerVerdict


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    """Утверждение, идентификаторы и цитата которого сверены с контекстом."""

    provisions: tuple[ProvisionId, ...]
    quote: str
    text: str


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    scenario: Scenario
    claims: tuple[GroundedClaim, ...]
    missing: tuple[str, ...]
    verdict: AnswerVerdict


# Ответ, когда из ответа модели взять нечего: рендерер даёт по нему отказ.
NO_DATA_ANSWER = GroundedAnswer(Scenario.INSUFFICIENT, (), (), AnswerVerdict.NONE)


@dataclass(frozen=True, slots=True)
class UnconfirmedAnswer:
    """Положения по вопросу есть, но ни одно утверждение о них не подтверждено.

    Не отказ: документ найден, и рендерер называет его ссылкой. Утверждений
    здесь нет — их снял верификатор.
    """

    # Положения, на которые опирались снятые утверждения; не пусто.
    provisions: tuple[ProvisionId, ...]


# Текст этих ответов даёт шаблон рендерера: пустой ответ модели здесь законен.
SCENARIOS_ANSWERED_BY_TEMPLATE = frozenset({Scenario.OFF_TOPIC, Scenario.ROLE_ATTACK})


class ProblemKind(Enum):
    """Чем ответ модели не подошёл. Значение — строка для повторного запроса."""

    NO_REPLY = "прокси не вернул ответ"
    CONTENT_FILTERED = "ответ остановлен фильтром содержимого"
    TRUNCATED = "ответ оборван по длине: сократи утверждения, JSON должен закрыться"
    EMPTY = "ответ пуст"
    BOUNDARY_LEAK = "в ответе метка блоков данных: содержимое запроса не повторяй"
    NOT_JSON = "ответ не разбирается как JSON: нужен один объект без текста вокруг"
    MISSING_FIELD = "нет обязательного поля"
    EXTRA_FIELD = "лишнее поле: допустимы только поля из примера"
    UNKNOWN_VALUE = "значение не из списка допустимых"
    WRONG_TYPE = "неверный тип значения"
    NOTHING_STATED = "нет ни утверждений, ни перечня недостающего"
    NO_PROVISION = "у утверждения нет идентификатора положения"
    FOREIGN_PROVISION = "идентификатора нет среди блоков provision этого запроса"
    QUOTE_TOO_SHORT = "цитата слишком короткая: нужен фрагмент с ключевым условием"
    QUOTE_NOT_IN_PROVISION = (
        "цитаты нет в тексте положения из ids: скопируй фрагмент дословно"
    )
    NO_TEXT = "текст утверждения пуст"


@dataclass(frozen=True, slots=True)
class ReplyProblem:
    kind: ProblemKind
    # Путь из имён полей контракта («claims.quote»); пусто — ответ целиком.
    field: str


@dataclass(frozen=True, slots=True)
class ReplyReading:
    # None — ответ не разобран по схеме, взять из него нечего.
    grounded: GroundedAnswer | None
    problems: tuple[ReplyProblem, ...]


@dataclass(frozen=True, slots=True)
class Generation:
    answer: GroundedAnswer
    # Повтор здесь и перегенерация после верификатора делят один дополнительный
    # вызов на вопрос: если он израсходован, перегенерации уже не будет.
    extra_call_spent: bool
    # Что не прошло проверку в ответе, из которого собран результат.
    problems: tuple[ReplyProblem, ...]


# Тот же запрос с перечнем ошибок эти причины не устранит.
_NOT_WORTH_RETRY = frozenset({ProblemKind.NO_REPLY, ProblemKind.CONTENT_FILTERED})
# Типы ошибок pydantic, которым соответствует своя причина; остальные — тип значения.
_SCHEMA_PROBLEM_KINDS: Mapping[str, ProblemKind] = {
    "missing": ProblemKind.MISSING_FIELD,
    "extra_forbidden": ProblemKind.EXTRA_FIELD,
    "enum": ProblemKind.UNKNOWN_VALUE,
}
_INVALID_JSON = "json_invalid"
_CONTRACT_FIELDS = frozenset(ModelAnswer.model_fields) | frozenset(
    ModelClaim.model_fields
)
_CLAIM_IDS = "claims.ids"
_CLAIM_QUOTE = "claims.quote"
_CLAIM_TEXT = "claims.text"


def generate(
    model: JsonChatModel,
    turns: Sequence[ChatTurn],
    provisions: Sequence[AdmittedProvision],
    boundary: Boundary,
) -> Generation:
    """Ответ модели, сверенный с контекстом; при ошибках — один повтор запроса.

    После повтора утверждения, не прошедшие сверку, отбрасываются. Если брать
    нечего, результат — NO_DATA_ANSWER.
    """
    first = ask(model, turns, provisions, boundary)
    if not first.problems or _retry_is_pointless(first.problems):
        return Generation(first.grounded or NO_DATA_ANSWER, False, first.problems)
    corrected = correction_request(turns, first.problems, provisions)
    second = ask(model, corrected, provisions, boundary)
    # Неразобранный повтор не отменяет того, что прошло сверку в первом ответе.
    chosen = second if second.grounded is not None else first
    return Generation(chosen.grounded or NO_DATA_ANSWER, True, chosen.problems)


def ask(
    model: JsonChatModel,
    turns: Sequence[ChatTurn],
    provisions: Sequence[AdmittedProvision],
    boundary: Boundary,
) -> ReplyReading:
    """Один вызов модели без повтора; сбой прокси — причина, а не исключение."""
    try:
        completion = model.complete_json(turns)
    except ModelGatewayError:
        # Клиент прокси уже повторял запрос сам.
        return _unreadable(ProblemKind.NO_REPLY)
    return read_reply(completion, provisions, boundary)


def read_reply(
    completion: JsonCompletion,
    provisions: Sequence[AdmittedProvision],
    boundary: Boundary,
) -> ReplyReading:
    """Что в ответе модели прошло проверку и чем не подошло остальное."""
    if completion.finish is FinishReason.CONTENT_FILTER:
        return _unreadable(ProblemKind.CONTENT_FILTERED)
    if completion.finish is FinishReason.LENGTH:
        return _unreadable(ProblemKind.TRUNCATED)
    if not completion.content.strip():
        return _unreadable(ProblemKind.EMPTY)
    if mentions_boundary(completion.content, boundary):
        # Метку знает только запрос: её появление значит, что модель пересказывает
        # промпт, и остальному тексту ответа верить нельзя.
        return _unreadable(ProblemKind.BOUNDARY_LEAK)
    try:
        answer = ModelAnswer.model_validate_json(completion.content)
    except ValidationError as error:
        return ReplyReading(None, _schema_problems(error))
    return _checked_against_context(answer, provisions)


def correction_request(
    turns: Sequence[ChatTurn],
    problems: Sequence[ReplyProblem],
    provisions: Sequence[AdmittedProvision],
) -> list[ChatTurn]:
    """Тот же запрос, к последнему сообщению которого дописан перечень ошибок.

    Запрос обязан отличаться от первого: при нулевой температуре одинаковый
    запрос вернёт тот же ответ. В перечне только наш текст и идентификаторы
    положений контекста — строк из ответа модели в нём нет.
    """
    allowed = ", ".join(provision.id for provision in provisions)
    lines = "\n".join(
        f"- {_described(problem, allowed or _NO_PROVISIONS_IN_REQUEST)}"
        for problem in problems
    )
    *before, last = turns
    correction = ANSWER_CORRECTION_V1.format(problems=lines)
    return [*before, ChatTurn(last.role, f"{last.content}\n{correction}")]


def _retry_is_pointless(problems: Sequence[ReplyProblem]) -> bool:
    return any(problem.kind in _NOT_WORTH_RETRY for problem in problems)


def _unreadable(kind: ProblemKind) -> ReplyReading:
    return ReplyReading(None, (ReplyProblem(kind, ""),))


def _schema_problems(error: ValidationError) -> tuple[ReplyProblem, ...]:
    problems: list[ReplyProblem] = []
    for item in error.errors(
        include_url=False, include_context=False, include_input=False
    ):
        if item["type"] == _INVALID_JSON:
            return (ReplyProblem(ProblemKind.NOT_JSON, ""),)
        kind = _SCHEMA_PROBLEM_KINDS.get(item["type"], ProblemKind.WRONG_TYPE)
        # Имя лишнего поля придумала модель; в путь идут только имена контракта.
        path = [part for part in item["loc"] if part in _CONTRACT_FIELDS]
        problems.append(ReplyProblem(kind, ".".join(str(part) for part in path)))
    return _distinct(problems)


def _checked_against_context(
    answer: ModelAnswer, provisions: Sequence[AdmittedProvision]
) -> ReplyReading:
    texts = {provision.id: comparison_key(provision.text) for provision in provisions}
    claims: list[GroundedClaim] = []
    problems: list[ReplyProblem] = []
    for claim in answer.claims:
        found = _claim_problems(claim, texts)
        problems.extend(found)
        if not found:
            claims.append(_grounded(claim))
    missing = tuple(item.strip() for item in answer.missing if item.strip())
    if answer.claims and not claims:
        return ReplyReading(NO_DATA_ANSWER, _distinct(problems))
    if not claims and not missing:
        if answer.scenario in SCENARIOS_ANSWERED_BY_TEMPLATE:
            return ReplyReading(_without_claims(answer.scenario), ())
        return ReplyReading(
            NO_DATA_ANSWER, (ReplyProblem(ProblemKind.NOTHING_STATED, ""),)
        )
    # «Да» и «нет» опираются на все утверждения ответа: без любого из них и без
    # утверждений вообще прямой ответ остаётся неподтверждённым.
    intact = bool(claims) and len(claims) == len(answer.claims)
    verdict = answer.verdict if intact else AnswerVerdict.NONE
    grounded = GroundedAnswer(answer.scenario, tuple(claims), missing, verdict)
    return ReplyReading(grounded, _distinct(problems))


def _claim_problems(
    claim: ModelClaim, texts: Mapping[ProvisionId, str]
) -> list[ReplyProblem]:
    problems: list[ReplyProblem] = []
    cited = _cited(claim)
    known = all(provision in texts for provision in cited)
    if not cited:
        problems.append(ReplyProblem(ProblemKind.NO_PROVISION, _CLAIM_IDS))
    elif not known:
        problems.append(ReplyProblem(ProblemKind.FOREIGN_PROVISION, _CLAIM_IDS))
    # Модель может поменять регистр, кавычки и пробелы: сверяем ключи сравнения.
    quote = comparison_key(claim.quote)
    if len(quote) < MIN_QUOTE_LENGTH:
        problems.append(ReplyProblem(ProblemKind.QUOTE_TOO_SHORT, _CLAIM_QUOTE))
    elif cited and known and not any(quote in texts[one] for one in cited):
        problems.append(ReplyProblem(ProblemKind.QUOTE_NOT_IN_PROVISION, _CLAIM_QUOTE))
    if not claim.text.strip():
        problems.append(ReplyProblem(ProblemKind.NO_TEXT, _CLAIM_TEXT))
    return problems


def _grounded(claim: ModelClaim) -> GroundedClaim:
    return GroundedClaim(_cited(claim), claim.quote.strip(), claim.text.strip())


def _cited(claim: ModelClaim) -> tuple[ProvisionId, ...]:
    return tuple(dict.fromkeys(ProvisionId(one) for one in claim.ids))


def _without_claims(scenario: Scenario) -> GroundedAnswer:
    return GroundedAnswer(scenario, (), (), AnswerVerdict.NONE)


def _described(problem: ReplyProblem, allowed: str) -> str:
    text = problem.kind.value
    if problem.kind is ProblemKind.FOREIGN_PROVISION:
        text = f"{text}; допустимые: {allowed}"
    return f"{problem.field}: {text}" if problem.field else text


def _distinct(problems: Sequence[ReplyProblem]) -> tuple[ReplyProblem, ...]:
    return tuple(dict.fromkeys(problems))
