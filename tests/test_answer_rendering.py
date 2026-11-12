from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence

import pytest

import lex_answer_rendering
from lex_answer_contract import (
    NO_DATA_ANSWER,
    AnswerVerdict,
    GroundedAnswer,
    GroundedClaim,
    generate,
)
from lex_answer_rendering import (
    CLARIFICATION_REQUEST,
    ILLEGAL_LEAD,
    MAX_ANSWER_WORDS,
    OFF_TOPIC_REPLY,
    REFUSAL_NO_DATA,
    ROLE_REPLY,
    render,
)
from lex_documents import DocumentId, ProvisionId, cite
from lex_errors import VerificationError
from lex_knowledge_base import KnowledgeBase, read_document
from lex_models import ChatTurn, FinishReason, JsonCompletion
from lex_prompt_boundary import Boundary
from lex_sanitizer import AdmittedProvision
from lex_scenarios import Scenario
from synthetic_base import SOURCES

BASE = KnowledgeBase(
    tuple(
        read_document(DocumentId(f"d{number:02}"), source)
        for number, source in enumerate(SOURCES, start=1)
    )
)
CONTRACT_TITLE = "Договор оказания услуг № У-03/2031"
MEMO_TITLE = "Служебная записка № ОК-4 от 20.04.2031"
PRICE = "d01.6"  # п. 2.1
PAYMENT = "d01.7"  # п. 2.2
RESPONSE = "d01.9"  # п. 3.2
SCHEDULE = "d02.5"  # абз. 1
LEAVE_PAY = "d03.2"  # ТК РФ, ст. 300, ч. 2

BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
TURNS = [ChatTurn("system", "Правила."), ChatTurn("user", "Вопрос.")]


def _claim(text: str, *provisions: str) -> GroundedClaim:
    return GroundedClaim(
        provisions=tuple(ProvisionId(one) for one in provisions),
        quote="цитата рендереру не нужна",
        text=text,
    )


def _answer(
    *claims: GroundedClaim,
    scenario: Scenario = Scenario.DIRECT,
    missing: Sequence[str] = (),
    verdict: AnswerVerdict = AnswerVerdict.NONE,
) -> GroundedAnswer:
    return GroundedAnswer(scenario, tuple(claims), tuple(missing), verdict)


def test_each_claim_gets_reference_and_document_is_named_once() -> None:
    answer = _answer(
        _claim("Стоимость услуг — 80 000 рублей в месяц.", PRICE),
        _claim("Оплата — в течение 10 рабочих дней с даты подписания акта.", PAYMENT),
        _claim("На претензию отвечают в течение 15 календарных дней.", RESPONSE),
    )
    assert render(answer, BASE) == (
        f"Стоимость услуг — 80 000 рублей в месяц ({CONTRACT_TITLE}, п. 2.1). "
        "Оплата — в течение 10 рабочих дней с даты подписания акта (п. 2.2). "
        "На претензию отвечают в течение 15 календарных дней (п. 3.2)."
    )


def test_verdict_goes_first_and_single_provision_is_named_as_source() -> None:
    answer = _answer(
        _claim("Учебный отпуск оплачивается по среднему заработку.", LEAVE_PAY),
        verdict=AnswerVerdict.YES,
    )
    assert render(answer, BASE) == (
        "Да. Учебный отпуск оплачивается по среднему заработку. "
        "Источник: ТК РФ, ст. 300, ч. 2."
    )


def test_negative_verdict_goes_first() -> None:
    answer = _answer(
        _claim("Срок оплаты — 10 рабочих дней.", PAYMENT), verdict=AnswerVerdict.NO
    )
    assert render(answer, BASE).startswith("Нет. Срок оплаты")


def test_claims_with_the_same_provision_share_one_reference() -> None:
    answer = _answer(
        _claim("Оплата производится после подписания акта.", PAYMENT),
        _claim("Срок оплаты — 10 рабочих дней.", PAYMENT),
        _claim("На претензию отвечают в течение 15 календарных дней.", RESPONSE),
    )
    assert render(answer, BASE) == (
        "Оплата производится после подписания акта. "
        f"Срок оплаты — 10 рабочих дней ({CONTRACT_TITLE}, п. 2.2). "
        "На претензию отвечают в течение 15 календарных дней (п. 3.2)."
    )


def test_provisions_of_one_document_are_folded_under_its_title() -> None:
    answer = _answer(
        _claim(
            "Услуги стоят 80 000 рублей в месяц и оплачиваются по акту.", PRICE, PAYMENT
        )
    )
    assert render(answer, BASE).endswith(f"Источник: {CONTRACT_TITLE}, п. 2.1, п. 2.2.")


def test_provisions_of_two_documents_are_separated_in_one_reference() -> None:
    answer = _answer(
        _claim("Оплата по акту, отпуск оплачивается по среднему.", PAYMENT, LEAVE_PAY)
    )
    assert render(answer, BASE).endswith(
        f"Источник: {CONTRACT_TITLE}, п. 2.2; ТК РФ, ст. 300, ч. 2."
    )


def test_document_is_named_again_after_another_document() -> None:
    answer = _answer(
        _claim("Срок оплаты — 10 рабочих дней.", PAYMENT),
        _claim("График отпусков утверждается до 15 декабря.", SCHEDULE),
        _claim("На претензию отвечают в течение 15 календарных дней.", RESPONSE),
    )
    assert render(answer, BASE) == (
        f"Срок оплаты — 10 рабочих дней ({CONTRACT_TITLE}, п. 2.2). "
        f"График отпусков утверждается до 15 декабря ({MEMO_TITLE}, абз. 1). "
        "На претензию отвечают в течение 15 календарных дней "
        f"({CONTRACT_TITLE}, п. 3.2)."
    )


def test_claim_without_full_stop_still_ends_with_one() -> None:
    answer = _answer(
        _claim("Срок оплаты — 10 рабочих дней", PAYMENT),
        _claim("На претензию отвечают 15 календарных дней", RESPONSE),
    )
    assert render(answer, BASE) == (
        f"Срок оплаты — 10 рабочих дней ({CONTRACT_TITLE}, п. 2.2). "
        "На претензию отвечают 15 календарных дней (п. 3.2)."
    )


def test_refusal_names_missing_topic_and_declines_to_answer_from_memory() -> None:
    answer = _answer(
        scenario=Scenario.INSUFFICIENT, missing=["пособиях по беременности и родам"]
    )
    assert render(answer, BASE) == (
        "В базе знаний нет документов о пособиях по беременности и родам, "
        "поэтому ответить по документам я не могу. По памяти отвечать не буду."
    )


def test_safe_answer_is_a_refusal_without_topic() -> None:
    assert render(NO_DATA_ANSWER, BASE) == REFUSAL_NO_DATA


def test_partial_answer_ends_with_what_is_missing() -> None:
    answer = _answer(
        _claim("Стоимость услуг — 80 000 рублей в месяц, а не 100 000.", PRICE),
        scenario=Scenario.FALSE_PREMISE,
        missing=["увеличении стоимости услуг"],
    )
    assert render(answer, BASE) == (
        "Стоимость услуг — 80 000 рублей в месяц, а не 100 000. "
        f"Источник: {CONTRACT_TITLE}, п. 2.1. "
        "В базе знаний нет документов об увеличении стоимости услуг."
    )


@pytest.mark.parametrize(
    ("topic", "phrase"),
    [
        ("пособиях", "о пособиях"),
        ("увеличении платы", "об увеличении платы"),
        ("отпусках.", "об отпусках"),
        ("единой ставке", "о единой ставке"),
        ("юридических лицах", "о юридических лицах"),
        ("«Альфа-Трейд»", "об «Альфа-Трейд»"),
        ("о пособиях", "о пособиях"),
        ("Об условиях поставки", "об условиях поставки"),
        ("обо всех доплатах", "обо всех доплатах"),
        ("оборудовании", "об оборудовании"),
    ],
)
def test_missing_topic_gets_preposition_by_first_sound(topic: str, phrase: str) -> None:
    text = render(_answer(scenario=Scenario.INSUFFICIENT, missing=[topic]), BASE)
    assert f"нет документов {phrase}, поэтому" in text


def test_several_missing_topics_are_enumerated() -> None:
    answer = _answer(
        scenario=Scenario.INSUFFICIENT, missing=["пособиях", "отпусках", "увольнении"]
    )
    assert "нет документов о пособиях, об отпусках и об увольнении," in render(
        answer, BASE
    )


def test_off_topic_reply_is_a_template_whatever_the_model_returned() -> None:
    answer = _answer(
        _claim("Срок оплаты — 10 рабочих дней.", PAYMENT), scenario=Scenario.OFF_TOPIC
    )
    assert render(answer, BASE) == OFF_TOPIC_REPLY


def test_role_attack_without_claims_gets_role_reply() -> None:
    assert render(_answer(scenario=Scenario.ROLE_ATTACK), BASE) == ROLE_REPLY


def test_role_attack_with_claims_is_answered_as_usual() -> None:
    answer = _answer(
        _claim("Срок оплаты — 10 рабочих дней.", PAYMENT), scenario=Scenario.ROLE_ATTACK
    )
    assert render(answer, BASE) == (
        f"Срок оплаты — 10 рабочих дней. Источник: {CONTRACT_TITLE}, п. 2.2."
    )


def test_illegal_request_is_declined_before_the_norm() -> None:
    answer = _answer(
        _claim("Стороны не разглашают условия договора третьим лицам.", "d01.10"),
        scenario=Scenario.ILLEGAL,
        verdict=AnswerVerdict.NO,
    )
    assert render(answer, BASE) == (
        f"{ILLEGAL_LEAD} Стороны не разглашают условия договора третьим лицам. "
        f"Источник: {CONTRACT_TITLE}, п. 3.3."
    )


def test_ambiguous_question_without_claims_asks_to_clarify() -> None:
    answer = _answer(
        scenario=Scenario.AMBIGUOUS,
        missing=["какой договор с «Гранитом» имеется в виду?"],
    )
    assert render(answer, BASE) == CLARIFICATION_REQUEST.format(
        details="какой договор с «Гранитом» имеется в виду"
    )


def test_ambiguous_question_answered_by_variants_names_each_document() -> None:
    answer = _answer(
        _claim("По договору с ООО «Гранит» срок оплаты — 10 рабочих дней.", PAYMENT),
        _claim("По записке отдела кадров график утверждается до 15 декабря.", SCHEDULE),
        scenario=Scenario.AMBIGUOUS,
    )
    text = render(answer, BASE)
    assert f"({CONTRACT_TITLE}, п. 2.2)" in text
    assert f"({MEMO_TITLE}, абз. 1)" in text


def test_long_answer_loses_claims_from_the_end() -> None:
    filler = " ".join(["слово"] * 60)
    answer = _answer(
        _claim(f"Первое {filler}.", PRICE),
        _claim(f"Второе {filler}.", PAYMENT),
        _claim(f"Третье {filler}.", RESPONSE),
    )
    text = render(answer, BASE)
    assert len(text.split()) <= MAX_ANSWER_WORDS
    assert "Первое" in text
    assert "Второе" in text
    assert "Третье" not in text
    assert "п. 3.2" not in text


def test_provision_missing_from_base_is_an_error_not_a_reference() -> None:
    with pytest.raises(VerificationError, match=r"d09\.1"):
        render(_answer(_claim("Срок — 10 дней.", "d09.1")), BASE)


def test_templates_state_no_numbers_articles_or_clauses() -> None:
    templates = [
        value
        for name, value in vars(lex_answer_rendering).items()
        if name.isupper() and not name.startswith("_") and isinstance(value, str)
    ]
    templates.extend(lex_answer_rendering.VERDICT_LEADS.values())
    assert len(templates) >= 9
    for template in templates:
        assert re.search(r"\d|ст\.|п\.|закон|кодекс", template) is None, template


class _RepeatingModel:
    """На первый запрос и на повтор отвечает одним и тем же."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete_json(self, turns: Sequence[ChatTurn]) -> JsonCompletion:
        return JsonCompletion(content=self._reply, finish=FinishReason.STOP)


EVERY_PROVISION = [
    (document, provision)
    for document in BASE.documents
    for provision in document.provisions
]
TEXT_BY_ID = {
    document.provision_id(provision): provision.text
    for document, provision in EVERY_PROVISION
}
NOT_IN_BASE = ["d99.1", "x1.2", "d01.70", "п. 3.1"]
CLAIM_TEXTS = ["Первое утверждение.", "Второе утверждение.", "Третье утверждение."]


def _random_reply(rng: random.Random, context: Sequence[str]) -> str:
    claims = []
    for text in rng.sample(CLAIM_TEXTS, k=rng.randint(1, len(CLAIM_TEXTS))):
        pool = context if rng.random() < 0.5 else [*TEXT_BY_ID, *NOT_IN_BASE]
        ids = rng.sample(pool, k=min(len(pool), rng.randint(1, 2)))
        # Цитата настоящая: отсечь чужое положение может только белый список.
        quoted = TEXT_BY_ID.get(ProvisionId(ids[0]), "цитата из положения вне базы")
        claims.append({"ids": ids, "quote": quoted[:60], "text": text})
    reply = {"scenario": "direct", "claims": claims, "missing": [], "verdict": "none"}
    return json.dumps(reply, ensure_ascii=False)


def test_rendered_text_never_refers_to_provision_outside_context() -> None:
    rng = random.Random(53)
    answers_with_references = 0
    for _ in range(400):
        chosen = rng.sample(EVERY_PROVISION, k=rng.randint(1, 6))
        context = [
            AdmittedProvision(
                id=document.provision_id(provision),
                heading=cite(document, provision),
                text=provision.text,
            )
            for document, provision in chosen
        ]
        allowed = {provision.id for provision in context}
        allowed_references = {
            cite(document, provision) for document, provision in chosen
        }
        allowed_labels = {provision.label for _, provision in chosen}
        reply = _random_reply(rng, sorted(allowed))

        generation = generate(_RepeatingModel(reply), TURNS, context, BOUNDARY)
        text = render(generation.answer, BASE)

        cited = {one for claim in generation.answer.claims for one in claim.provisions}
        assert cited <= allowed
        answers_with_references += bool(cited)
        for document, provision in EVERY_PROVISION:
            if cite(document, provision) not in allowed_references:
                assert cite(document, provision) not in text
            if provision.label not in allowed_labels:
                assert re.search(rf"{re.escape(provision.label)}(?!\d)", text) is None
    # Иначе проверка выше ничего не доказывает: ссылок в текстах не было.
    assert answers_with_references > 100
