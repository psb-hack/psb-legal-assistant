"""Шаблоны атак из eval/attacks.yaml в каждом месте, куда попадают данные.

Проверяется сборка контекста, а не поведение модели: после подстановки шаблона
в вопрос, историю и текст положения запрос проходит проверку границы.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lex_documents import ProvisionId
from lex_prompt_boundary import (
    Boundary,
    PromptRules,
    assemble_prompt,
    boundary_violations,
)
from lex_role_markers import find_role_markers
from lex_sanitizer import AdmittedProvision

ATTACKS = Path(__file__).resolve().parents[1] / "eval" / "attacks.yaml"
SECTIONS = yaml.safe_load(ATTACKS.read_text(encoding="utf-8"))
BOUNDARY = Boundary("9f2c4e6a8b0d1f35")
RULES = PromptRules(
    system="Отвечай только по блокам provision.", reminder="Ответ — JSON."
)
QUESTION = "Какой срок оплаты по договору № У-03/2031?"
HEADING = "Договор оказания услуг № У-03/2031, п. 2.2"
SUBSTITUTIONS = {
    "question": QUESTION,
    "attribute": "срок оплаты",
    "object": "договор оказания услуг № У-03/2031",
    "wrong_value": "90 календарных дней",
}
# Шаблоны, в которых есть разметка чата или метка роли: по ним видно, что
# проверка не проходит впустую.
MARKED_TEMPLATES = {
    "kb_injection/role_override",
    "user_attack/fake_system",
    "user_attack/chat_markup",
    "fake_history/wrong_assistant_fact",
    "fake_history/assistant_agreed",
    "fake_history/nested_system",
    "fake_history/user_insists",
}


def _attacks() -> dict[str, tuple[str, ...]]:
    """Имя шаблона → его строки: текст, части вставки или реплики истории."""
    found: dict[str, tuple[str, ...]] = {}
    for section in ("kb_injection", "kb_misinformation", "user_attack", "fake_history"):
        for template in SECTIONS[section]["templates"]:
            lines = template.get("turns") or template.get("parts") or [template["text"]]
            found[f"{section}/{template['id']}"] = tuple(
                line.format(**SUBSTITUTIONS) for line in lines
            )
    return found


ATTACK_LINES = _attacks()


def _provision(text: str) -> AdmittedProvision:
    return AdmittedProvision(ProvisionId("d01.7"), HEADING, text)


def test_every_template_of_the_attack_sections_is_read() -> None:
    assert len(ATTACK_LINES) == sum(
        len(SECTIONS[section]["templates"])
        for section in (
            "kb_injection",
            "kb_misinformation",
            "user_attack",
            "fake_history",
        )
    )
    assert len(SECTIONS["user_attack"]["templates"]) >= 9
    assert len(SECTIONS["fake_history"]["templates"]) >= 4


@pytest.mark.parametrize("name", sorted(ATTACK_LINES))
def test_context_with_the_attack_passes_the_boundary_check(name: str) -> None:
    lines = ATTACK_LINES[name]
    joined = "\n".join(lines)
    placements = (
        ((), (), joined),
        ((), lines, QUESTION),
        ((), (joined,), QUESTION),
        ((_provision(joined),), (), QUESTION),
        ((_provision(joined),), lines, joined),
    )
    for provisions, history, question in placements:
        turns = assemble_prompt(RULES, provisions, history, question, BOUNDARY)
        assert turns[0].content == RULES.system
        assert boundary_violations(turns, RULES, BOUNDARY) == ()


def test_templates_with_markers_are_exactly_the_known_ones() -> None:
    marked = {
        name
        for name, lines in ATTACK_LINES.items()
        if find_role_markers("\n".join(lines))
    }
    assert marked == MARKED_TEMPLATES


@pytest.mark.parametrize("name", sorted(MARKED_TEMPLATES))
def test_unneutralized_attack_inside_a_block_fails_the_check(name: str) -> None:
    joined = "\n".join(ATTACK_LINES[name])
    system, user = assemble_prompt(RULES, (), (), QUESTION, BOUNDARY)
    raw = type(user)(user.role, user.content.replace(QUESTION, joined, 1))
    assert boundary_violations([system, raw], RULES, BOUNDARY) != ()
