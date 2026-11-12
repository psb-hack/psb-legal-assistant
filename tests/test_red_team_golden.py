"""Защита до модели на настоящей базе: исходы проверок уровня контекста по
демо-вопросам. Архив в репозиторий не входит; полный набор кейсов считает
scripts/red_team.py --context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from lex_knowledge_base import read_archive
from red_team_context import CONTEXT_CHECKS, assembler
from red_team_invariants import Verdict
from red_team_mutations import load_cases, registry
from red_team_run import report, run

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "knowledge-base.zip"
GOLDEN = Path(__file__).resolve().parent / "golden" / "red_team.yaml"
EXPECTED = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


def measured() -> dict[str, Any]:
    """Исходы в виде, в котором они лежат в tests/golden/red_team.yaml."""
    cases = [case for case in load_cases() if case.id.startswith("demo-")]
    trials = run(
        cases,
        read_archive(ARCHIVE),
        assembler(EXPECTED["today"]),
        CONTEXT_CHECKS,
        registry(),
        EXPECTED["seed"],
    )
    counted = report(trials)
    return {
        "mutated_cases": {
            name: row["mutated_cases"] for name, row in counted["categories"].items()
        },
        # Проверка: выдержано, нарушено, неопределённо.
        "checks": {
            name: {
                invariant: [by_verdict[item.value] for item in Verdict]
                for invariant, by_verdict in row["invariants"].items()
            }
            for name, row in counted["categories"].items()
        },
        "violated_by_mutation": {
            name: row["violated"]
            for name, row in counted["mutations"].items()
            if row["violated"]
        },
    }


def test_context_level_outcomes_on_demo_cases_are_as_recorded() -> None:
    found = measured()
    assert found == {key: EXPECTED[key] for key in found}
