"""LLM-классификатор для фрагментов со слабым сигналом санитайзера.

Классификатор может добавить карантин, но не снять его: фрагменты с сильным
сигналом детекторов он не читает.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, ValidationError

from lex_detection import Signal, SignalStrength
from lex_errors import ModelGatewayError
from lex_injection_prompt import Addressee, classifier_turns
from lex_models import ChatModel, ChatTurn
from lex_normalization import clean_text, comparison_key
from lex_sanitizer import Decision, Screening, Verdict

DETECTOR_NAME = "классификатор"
FEATURE_INSTRUCTION = "указание ИИ-системе"
FEATURE_NO_INSTRUCTION = "указаний ИИ-системе нет"
FEATURE_NO_ANSWER = "классификатор не ответил"

# Первый ответ и один повтор с описанием ошибки.
_ATTEMPTS = 2
_MARKER_BYTES = 8
_REJECTION = (
    "Ответ отклонён: {problem}. Верни только JSON-объект по схеме из правил, "
    "без пояснений."
)
_NOT_BY_SCHEMA = "это не JSON-объект по схеме"
_NO_QUOTE = "указание найдено, но цитаты нет"
_QUOTE_NOT_IN_FRAGMENT = "цитаты нет во фрагменте"


class InjectionVerdict(BaseModel):
    """Контракт ответа модели: лишнее поле или неизвестный адресат — ошибка."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    addressee: Addressee
    contains_instructions_for_ai: bool
    quote: str

    @property
    def is_instruction(self) -> bool:
        return (
            self.contains_instructions_for_ai or self.addressee is Addressee.AI_SYSTEM
        )


def random_marker() -> str:
    """Маркер на один вызов: автор вставки не может знать его заранее."""
    return secrets.token_hex(_MARKER_BYTES)


def classify(fragment: str, model: ChatModel, marker: str) -> InjectionVerdict | None:
    """Ответ модели по схеме; None — ответа нет и после повтора."""
    turns = list(classifier_turns(fragment, marker))
    for _ in range(_ATTEMPTS):
        try:
            reply = model.complete(turns).content
        except ModelGatewayError:
            # Клиент прокси уже повторял запрос сам.
            return None
        verdict = _checked(reply, fragment)
        if isinstance(verdict, InjectionVerdict):
            return verdict
        turns.append(ChatTurn("assistant", reply))
        turns.append(ChatTurn("user", _REJECTION.format(problem=verdict)))
    return None


def review_suspects(
    screening: Screening, model: ChatModel, next_marker: Callable[[], str]
) -> Screening:
    """Решения по фрагментам со слабым сигналом; остальные записи не меняются."""
    return Screening(
        tuple(
            _reviewed(decision, model, next_marker())
            if decision.verdict is Verdict.SUSPECT
            else decision
            for decision in screening.decisions
        )
    )


def _reviewed(decision: Decision, model: ChatModel, marker: str) -> Decision:
    verdict = classify(clean_text(decision.original), model, marker)
    if verdict is None:
        # Слабый сигнал уже есть, а проверить его нечем: фрагмент не пропускаем.
        return _with_signal(
            decision, Verdict.QUARANTINED, SignalStrength.STRONG, (FEATURE_NO_ANSWER,)
        )
    addressee = f"адресат: {verdict.addressee.value}"
    if verdict.is_instruction:
        return _with_signal(
            decision,
            Verdict.QUARANTINED,
            SignalStrength.STRONG,
            (FEATURE_INSTRUCTION, addressee),
        )
    return _with_signal(
        decision,
        Verdict.CLEARED,
        SignalStrength.NONE,
        (FEATURE_NO_INSTRUCTION, addressee),
    )


def _with_signal(
    decision: Decision,
    verdict: Verdict,
    strength: SignalStrength,
    features: tuple[str, ...],
) -> Decision:
    signal = Signal(DETECTOR_NAME, strength, features)
    return replace(decision, verdict=verdict, signals=(*decision.signals, signal))


def _checked(reply: str, fragment: str) -> InjectionVerdict | str:
    """Ответ по схеме или описание того, чем он не подошёл."""
    try:
        verdict = InjectionVerdict.model_validate_json(reply)
    except ValidationError:
        return _NOT_BY_SCHEMA
    if verdict.is_instruction and not verdict.quote.strip():
        return _NO_QUOTE
    # Модель может поменять регистр, кавычки и пробелы: сверяем ключи сравнения.
    if comparison_key(verdict.quote) not in comparison_key(fragment):
        return _QUOTE_NOT_IN_FRAGMENT
    return verdict
