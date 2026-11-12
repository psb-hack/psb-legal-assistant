from __future__ import annotations

import re
from collections.abc import Sequence

import pytest

import lex_answer_rendering
from lex_answer_contract import (
    NO_DATA_ANSWER,
    AnswerVerdict,
    ModelAnswer,
    ModelClaim,
    ProblemKind,
    generate,
    read_reply,
)
from lex_answer_rendering import render
from lex_documents import DocumentId, ProvisionId, cite
from lex_knowledge_base import KnowledgeBase, read_document
from lex_models import ChatTurn, FinishReason, JsonCompletion
from lex_normalization import comparison_key
from lex_prompt_boundary import (
    DATA_LEGEND_V1,
    Boundary,
    assemble_prompt,
    boundary_violations,
)
from lex_prompts import (
    ANSWER_EXAMPLES_V1,
    ANSWER_PROMPT_V1,
    ANSWER_REMINDER_V1,
    ANSWER_RULES_V1,
    PromptExample,
)
from lex_sanitizer import AdmittedProvision
from lex_scenarios import Scenario
from prompt_conformance import (
    LEADING_PREPOSITION,
    REFERENCE_IN_TEXT,
    SECOND_SENTENCE,
    rule_breaches,
)
from synthetic_base import SOURCES

BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
BASE = KnowledgeBase(
    tuple(
        read_document(DocumentId(f"d{number:02}"), source)
        for number, source in enumerate(SOURCES, start=1)
    )
)
CONTEXT = tuple(
    AdmittedProvision(
        id=document.provision_id(provision),
        heading=cite(document, provision),
        text=provision.text,
    )
    for document in BASE.documents
    for provision in document.provisions
)
PRICE = "d01.6"  # п. 2.1
PAYMENT = "d01.7"  # п. 2.2
RESPONSE = "d01.9"  # п. 3.2

MAX_RULES = 10
MAX_EXAMPLES = 4
# «1. Факты бери только…»
RULE_NUMBER = re.compile(r"^(\d+)\. ", re.MULTILINE)
# «x1.4»: у положений базы идентификатор начинается с «d»
EXAMPLE_ID = re.compile(r"x\d+\.\d+")
# «ст. 70», «п. 3.1», «№ 7», «152-ФЗ», «Трудовой кодекс», «ООО», «по закону»
LEGAL_REFERENCE = re.compile(
    r"ст\.|п\.\s*\d|№|-фз|кодекс|\b[оаз]ао\b|\bооо\b|\bзакон(?:а|у|ом|е|ы|ов)?\b",
    re.IGNORECASE,
)
# Номера документов, стороны, акты и ключевые числа демо-базы организаторов.
DEMO_BASE_FACTS = (
    "А-08/2024",
    "П-17/2024",
    "ПТ-05/2023",
    "ЮО-7",
    "Северный Склад",
    "Альфа",
    "Меридиан",
    "Химки",
    "ТК РФ",
    "НК РФ",
    "КоАП",
    "152-ФЗ",
    "450 000",
    "500 000",
    "10 000 000",
    "15 000 000",
    "7%",
    "5-го числа",
    "5 рабочих дней",
    "45 календарных дней",
    "трёх рабочих дней",
    "шести месяцев",
    "24 месяца",
)
# Что код пишет сам: этих фраз модель писать не должна, и в правилах их нет.
CODE_TEMPLATES = (
    lex_answer_rendering.SOURCE_LINE,
    lex_answer_rendering.NO_DOCUMENTS_ABOUT,
    lex_answer_rendering.REFUSAL_ABOUT,
    lex_answer_rendering.REFUSAL_NO_DATA,
    lex_answer_rendering.UNCONFIRMED_IN_SOURCES,
    lex_answer_rendering.CLARIFICATION_REQUEST,
    lex_answer_rendering.ILLEGAL_LEAD,
    lex_answer_rendering.ROLE_REPLY,
    lex_answer_rendering.OFF_TOPIC_REPLY,
)
# Короче этого кусок шаблона — обычное слово, а не фраза кода.
MIN_TEMPLATE_PIECE = 10
PROMPT_TEXT = f"{ANSWER_PROMPT_V1.system}\n{ANSWER_PROMPT_V1.reminder}"
WITH_CLAIMS = [example for example in ANSWER_EXAMPLES_V1 if example.answer.claims]
REFUSALS = [example for example in ANSWER_EXAMPLES_V1 if not example.answer.claims]


class RecordingJsonModel:
    """Отвечает одним и тем же ответом и запоминает запросы."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.requests: list[list[ChatTurn]] = []

    def complete_json(self, turns: Sequence[ChatTurn]) -> JsonCompletion:
        self.requests.append(list(turns))
        return JsonCompletion(content=self._reply, finish=FinishReason.STOP)


def _example_replies() -> list[str]:
    """Строки JSON в том виде, в каком они стоят в system."""
    return [
        line for line in ANSWER_PROMPT_V1.system.split("\n") if line.startswith("{")
    ]


def _own_context(example: PromptExample) -> list[AdmittedProvision]:
    return [
        AdmittedProvision(ProvisionId(provision.id), provision.heading, provision.text)
        for provision in example.provisions
    ]


def _reply(
    claims: Sequence[tuple[str, str, str]],
    *,
    scenario: Scenario = Scenario.DIRECT,
    missing: Sequence[str] = (),
    verdict: AnswerVerdict = AnswerVerdict.NONE,
) -> ModelAnswer:
    return ModelAnswer(
        scenario=scenario,
        claims=tuple(
            ModelClaim(ids=(cited,), quote=quote, text=text)
            for cited, quote, text in claims
        ),
        missing=tuple(missing),
        verdict=verdict,
    )


def test_prompt_is_assembled_from_versioned_rules_examples_and_reminder() -> None:
    assert ANSWER_PROMPT_V1.system.startswith(ANSWER_RULES_V1)
    assert ANSWER_PROMPT_V1.reminder == ANSWER_REMINDER_V1
    assert [ModelAnswer.model_validate_json(line) for line in _example_replies()] == [
        example.answer for example in ANSWER_EXAMPLES_V1
    ]


def test_rules_are_numbered_in_order_and_fit_the_limit() -> None:
    numbers = [int(number) for number in RULE_NUMBER.findall(ANSWER_RULES_V1)]

    assert numbers == list(range(1, len(numbers) + 1))
    assert 0 < len(numbers) <= MAX_RULES


def test_json_mode_keyword_stands_in_rules_and_in_reminder() -> None:
    # JSON-режим прокси отклоняет запрос, в сообщениях которого нет слова «json».
    assert "JSON" in ANSWER_RULES_V1
    assert "JSON" in ANSWER_REMINDER_V1


def test_rules_name_every_field_scenario_and_verdict_of_the_schema() -> None:
    names = [
        *ModelAnswer.model_fields,
        *ModelClaim.model_fields,
        *(scenario.value for scenario in Scenario),
        *(verdict.value for verdict in AnswerVerdict),
    ]

    assert [name for name in names if name not in ANSWER_RULES_V1] == []


def test_examples_are_direct_partial_false_premise_and_single_refusal() -> None:
    answered = [example.answer for example in WITH_CLAIMS]

    assert len(ANSWER_EXAMPLES_V1) <= MAX_EXAMPLES
    assert len(REFUSALS) == 1
    assert REFUSALS[0].answer.missing
    assert {answer.scenario for answer in answered} == {
        Scenario.DIRECT,
        Scenario.INSUFFICIENT,
        Scenario.FALSE_PREMISE,
    }
    partial = [answer for answer in answered if answer.missing]
    assert [answer.scenario for answer in partial] == [Scenario.INSUFFICIENT]


@pytest.mark.parametrize("example", ANSWER_EXAMPLES_V1, ids=lambda e: e.question)
def test_example_is_valid_answer_to_its_own_provisions(example: PromptExample) -> None:
    completion = JsonCompletion(
        content=example.answer.model_dump_json(), finish=FinishReason.STOP
    )

    reading = read_reply(completion, _own_context(example), BOUNDARY)

    assert reading.problems == ()
    assert reading.grounded is not None
    assert len(reading.grounded.claims) == len(example.answer.claims)


@pytest.mark.parametrize("example", ANSWER_EXAMPLES_V1, ids=lambda e: e.question)
def test_example_follows_the_rules_it_illustrates(example: PromptExample) -> None:
    assert rule_breaches(example.answer) == []


def test_example_identifiers_differ_in_form_from_identifiers_of_the_base() -> None:
    ids = [
        provision.id
        for example in ANSWER_EXAMPLES_V1
        for provision in example.provisions
    ]

    assert [one for one in ids if EXAMPLE_ID.fullmatch(one) is None] == []
    assert len(set(ids)) == len(ids)
    assert not {provision.id for provision in CONTEXT} & set(ids)


def test_rules_carry_no_numbers_except_their_own_numbering() -> None:
    rules_count = len(RULE_NUMBER.findall(ANSWER_RULES_V1))
    numbers = re.findall(r"\d+", ANSWER_RULES_V1 + ANSWER_REMINDER_V1)

    assert [n for n in numbers if not 1 <= int(n) <= rules_count] == []


@pytest.mark.parametrize(
    "text",
    [
        "по ст. 70",
        "п. 3.1",
        "договор № 7",
        "152-ФЗ",
        "Трудовой кодекс",
        "ООО",
        "по закону",
    ],
)
def test_legal_reference_pattern_finds_norms_and_documents(text: str) -> None:
    assert LEGAL_REFERENCE.search(text)


@pytest.mark.parametrize(
    "text", ["законно ли действие", "законченное предложение", "номера статей"]
)
def test_legal_reference_pattern_passes_words_of_the_rules(text: str) -> None:
    assert LEGAL_REFERENCE.search(text) is None


def test_rules_and_reminder_state_no_norms_and_name_no_documents() -> None:
    assert LEGAL_REFERENCE.findall(ANSWER_RULES_V1 + ANSWER_REMINDER_V1) == []


def test_prompt_carries_no_fact_of_the_demo_base() -> None:
    prompt = comparison_key(PROMPT_TEXT)

    assert [fact for fact in DEMO_BASE_FACTS if comparison_key(fact) in prompt] == []


def test_prompt_does_not_dictate_phrases_that_code_writes() -> None:
    prompt = comparison_key(PROMPT_TEXT)
    pieces = [
        piece.strip()
        for template in CODE_TEMPLATES
        for piece in re.split(r"\{\w+\}", template)
        if len(piece.strip()) >= MIN_TEMPLATE_PIECE
    ]

    assert pieces
    assert [piece for piece in pieces if comparison_key(piece) in prompt] == []


def test_system_does_not_repeat_the_legend_of_data_blocks() -> None:
    sentences = [
        sentence.strip()
        for sentence in DATA_LEGEND_V1.format(token=BOUNDARY.token).split(". ")
    ]

    assert [one for one in sentences if one in ANSWER_PROMPT_V1.system] == []


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("Плата вносится по п. 3.1.", REFERENCE_IN_TEXT),
        ("Это следует из статьи 9.", REFERENCE_IN_TEXT),
        ("По договору № А-1 плата иная.", REFERENCE_IN_TEXT),
        ("Плата — 5 рублей. Вносится ежемесячно.", SECOND_SENTENCE),
        ("о пособиях", LEADING_PREPOSITION),
        ("Об отпуске", LEADING_PREPOSITION),
    ],
)
def test_conformance_patterns_find_what_rules_forbid(
    text: str, pattern: re.Pattern[str]
) -> None:
    assert pattern.search(text)


def test_conformance_check_names_each_broken_rule() -> None:
    wordy = _reply(
        [
            (
                PRICE,
                "80 000 рублей в месяц",
                "Цена — 80 000 рублей (п. 2.1). Платят раз в месяц.",
            )
        ],
        missing=["о скидках"],
    )
    templated = _reply([], scenario=Scenario.OFF_TOPIC, missing=["погоде"])
    unsupported = _reply([], missing=["скидках"], verdict=AnswerVerdict.YES)

    assert [breach.split(":")[0] for breach in rule_breaches(wordy)] == [
        "правило 2",
        "правило 4",
        "правило 5",
    ]
    assert [breach.split(":")[0] for breach in rule_breaches(templated)] == [
        "правило 9"
    ]
    assert [breach.split(":")[0] for breach in rule_breaches(unsupported)] == [
        "правило 10"
    ]


def test_system_is_the_same_for_every_request() -> None:
    # Одинаковый system попадает в кэш префикса; метка и данные — в user.
    first = assemble_prompt(ANSWER_PROMPT_V1, CONTEXT[:2], (), "Вопрос один?", BOUNDARY)
    second = assemble_prompt(
        ANSWER_PROMPT_V1,
        CONTEXT[2:],
        ("Пользователь: реплика.",),
        "Вопрос другой?",
        Boundary("0a1b2c3d4e5f6071"),
    )

    assert first[0] == second[0] == ChatTurn("system", ANSWER_PROMPT_V1.system)


ANSWERS_BY_THE_RULES = [
    pytest.param(
        "Сколько стоят услуги и когда их оплачивать?",
        _reply(
            [
                (PRICE, "80 000 рублей в месяц", "Услуги стоят 80 000 рублей в месяц."),
                (
                    PAYMENT,
                    "в течение 10 рабочих дней с даты подписания акта",
                    "Оплатить их нужно в течение 10 рабочих дней с подписания акта.",
                ),
            ]
        ),
        [
            "80 000 рублей в месяц (Договор оказания услуг № У-03/2031, п. 2.1)",
            "(п. 2.2)",
        ],
        id="direct",
    ),
    pytest.param(
        "Какой срок ответа на претензию и какая неустойка за просрочку оплаты?",
        _reply(
            [
                (
                    RESPONSE,
                    "15 календарных дней с даты её получения",
                    "На претензию отвечают в течение 15 календарных дней.",
                )
            ],
            scenario=Scenario.INSUFFICIENT,
            missing=["неустойке за просрочку оплаты"],
        ),
        [
            "Источник: Договор оказания услуг № У-03/2031, п. 3.2.",
            "нет документов о неустойке",
        ],
        id="partial",
    ),
    pytest.param(
        "Почему услуги стоят 95 000 рублей в месяц?",
        _reply(
            [
                (
                    PRICE,
                    "Стоимость услуг составляет 80 000 рублей в месяц",
                    "Услуги стоят 80 000 рублей в месяц, а не 95 000.",
                )
            ],
            scenario=Scenario.FALSE_PREMISE,
        ),
        ["80 000 рублей в месяц, а не 95 000.", "Источник:"],
        id="false-premise",
    ),
    pytest.param(
        "Правда ли, что на претензию отвечают за 15 календарных дней?",
        _reply(
            [
                (
                    RESPONSE,
                    "Срок ответа на претензию — 15 календарных дней",
                    "Срок ответа на претензию — 15 календарных дней.",
                )
            ],
            verdict=AnswerVerdict.YES,
        ),
        ["Да. Срок ответа на претензию — 15 календарных дней"],
        id="verdict",
    ),
    pytest.param(
        "Какой размер командировочных?",
        _reply([], scenario=Scenario.INSUFFICIENT, missing=["командировочных"]),
        ["В базе знаний нет документов о командировочных, поэтому"],
        id="refusal",
    ),
]


@pytest.mark.parametrize(("question", "reply", "fragments"), ANSWERS_BY_THE_RULES)
def test_request_built_from_rules_leads_to_answer_in_contract_form(
    question: str, reply: ModelAnswer, fragments: list[str]
) -> None:
    assert rule_breaches(reply) == []
    model = RecordingJsonModel(reply.model_dump_json())
    turns = assemble_prompt(ANSWER_PROMPT_V1, CONTEXT, (), question, BOUNDARY)

    generation = generate(model, turns, CONTEXT, BOUNDARY)
    text = render(generation.answer, BASE)

    (request,) = model.requests
    system, user = request
    assert system == ChatTurn("system", ANSWER_PROMPT_V1.system)
    assert boundary_violations(request, ANSWER_PROMPT_V1, BOUNDARY) == ()
    assert [p.id for p in CONTEXT if f'id="{p.id}"' not in user.content] == []
    assert question in user.content
    assert user.content.endswith(ANSWER_REMINDER_V1)
    assert generation.problems == ()
    assert [fragment for fragment in fragments if fragment not in text] == []


@pytest.mark.parametrize("example", WITH_CLAIMS, ids=lambda e: e.question)
def test_example_returned_as_answer_is_rejected_by_whitelist(
    example: PromptExample,
) -> None:
    model = RecordingJsonModel(example.answer.model_dump_json())
    turns = assemble_prompt(ANSWER_PROMPT_V1, CONTEXT, (), example.question, BOUNDARY)

    generation = generate(model, turns, CONTEXT, BOUNDARY)

    assert generation.answer == NO_DATA_ANSWER
    assert generation.extra_call_spent is True
    assert ProblemKind.FOREIGN_PROVISION in {p.kind for p in generation.problems}
    assert len(model.requests) == 2


def test_copied_refusal_example_is_not_caught_by_whitelist() -> None:
    # В отказе нет идентификаторов, сверять нечего: скопированный пример даёт
    # отказ с чужой темой. Выдуманного факта в нём нет, но отказ лишний —
    # частоту таких ответов меряем на настоящей модели.
    (example,) = REFUSALS
    model = RecordingJsonModel(example.answer.model_dump_json())
    turns = assemble_prompt(
        ANSWER_PROMPT_V1, CONTEXT, (), "Сколько стоят услуги?", BOUNDARY
    )

    generation = generate(model, turns, CONTEXT, BOUNDARY)

    assert generation.problems == ()
    assert generation.answer.claims == ()
    assert generation.answer.missing == example.answer.missing
