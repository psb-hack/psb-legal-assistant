"""Состязательный и метаморфный набор одной командой (PSB-55).

    python scripts/red_team.py --context
    python scripts/red_team.py --answerer red_team_run:ideal_answerer

--context меряет защиту до модели: запрос собирается детерминированной частью
пайплайна, ответчик не нужен. --answerer принимает функцию «кейс и база → ответ»
в записи «модуль:имя». Код возврата 1 — есть нарушенные или неопределённые
проверки.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Код оценки лежит в tests/, ядро — в src/. pytest добавляет оба каталога в путь
# сам, сценарию приходится сделать это до импорта.
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--context", action="store_true")
    target.add_argument("--answerer", metavar="МОДУЛЬ:ИМЯ")
    parser.add_argument(
        "--archive", type=Path, default=REPO / "data" / "knowledge-base.zip"
    )
    parser.add_argument("--cases", type=Path, default=REPO / "eval" / "cases.yaml")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--out", type=Path, default=REPO / "dist" / "red-team" / "report.json"
    )
    options = parser.parse_args(arguments)

    knowledge_base = importlib.import_module("lex_knowledge_base")
    mutations = importlib.import_module("red_team_mutations")
    runner = importlib.import_module("red_team_run")
    if options.context:
        context = importlib.import_module("red_team_context")
        respond = context.assembler(options.today)
        checks = context.CONTEXT_CHECKS
    else:
        module, _, name = options.answerer.partition(":")
        respond = getattr(importlib.import_module(module), name)
        checks = importlib.import_module("red_team_invariants").INVARIANTS
    trials = runner.run(
        mutations.load_cases(options.cases),
        knowledge_base.read_archive(options.archive),
        respond,
        checks,
        mutations.registry(),
        options.seed,
        options.rounds,
    )
    counted = runner.report(trials)
    run_parameters = {
        "level": "context" if options.context else "answer",
        "answerer": options.answerer,
        "seed": options.seed,
        "rounds": options.rounds,
        "today": options.today.isoformat(),
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(
        json.dumps({"run": run_parameters, **counted}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(runner.summary(counted))
    print(f"\nОтчёт: {options.out}")
    total = counted["total"]
    return 1 if total["violated"] or total["undetermined"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
