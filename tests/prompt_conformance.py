"""Та часть правил ANSWER_RULES_V1, которую можно проверить кодом.

Этой проверкой сверяются примеры из самого промпта и ответы, составленные
вручную для тестов: пример, нарушающий правило, учит модель его нарушать.
"""

from __future__ import annotations

import re

from lex_answer_contract import AnswerVerdict, ModelAnswer
from lex_scenarios import Scenario

# Правило 2: «не больше пяти».
MAX_CLAIMS = 5

# «п. 3.1», «ст. 70», «пункта 4», «статьёй 9», «№ А-08/2024»
REFERENCE_IN_TEXT = re.compile(
    r"\b(?:п|пп|ст|ч|абз)\.\s*\d|\b(?:пункт|стать|часть|части)\w*\s+\d|№",
    re.IGNORECASE,
)
# «…включая НДС. Вносится ежемесячно…»
SECOND_SENTENCE = re.compile(r"[.!?…]\s+[А-ЯЁA-Z«]")
# «о пособиях», «Об отпуске», «обо всех доплатах»
LEADING_PREPOSITION = re.compile(r"(?:обо|об|о)\s", re.IGNORECASE)

_ANSWERED_BY_TEMPLATE = frozenset({Scenario.ROLE_ATTACK, Scenario.OFF_TOPIC})


def rule_breaches(answer: ModelAnswer) -> list[str]:
    """Какие правила промпта нарушает структурный ответ; пусто — ни одного."""
    breaches: list[str] = []
    if len(answer.claims) > MAX_CLAIMS:
        breaches.append("правило 2: утверждений больше пяти")
    for claim in answer.claims:
        if SECOND_SENTENCE.search(claim.text):
            breaches.append(
                f"правило 2: в утверждении не одно предложение: {claim.text}"
            )
        if REFERENCE_IN_TEXT.search(claim.text):
            breaches.append(f"правило 4: в тексте ссылка на положение: {claim.text}")
    if answer.scenario is not Scenario.AMBIGUOUS:
        breaches.extend(
            f"правило 5: тема с предлогом: {topic}"
            for topic in answer.missing
            if LEADING_PREPOSITION.match(topic)
        )
    if answer.scenario in _ANSWERED_BY_TEMPLATE and (answer.claims or answer.missing):
        breaches.append("правило 9: у этого сценария claims и missing пустые")
    if answer.verdict is not AnswerVerdict.NONE and not answer.claims:
        breaches.append("правило 10: вердикт без утверждений")
    return breaches
