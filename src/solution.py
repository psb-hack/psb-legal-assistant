"""Точка входа сабмита: настройки с края, запись submission.csv."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lex_errors import LexError
from lex_settings import Settings
from lex_submission import load_questions, placeholder_answer, write_submission


def main(environ: dict[str, str] | None = None, cwd: Path | None = None) -> int:
    env = os.environ if environ is None else environ
    workdir = Path.cwd() if cwd is None else cwd
    settings = Settings.from_environ(env)
    questions_path = _questions_path(settings.data_dir)
    out = workdir / "submission.csv"
    try:
        questions = load_questions(questions_path)
        answers = {item.id: placeholder_answer(item) for item in questions}
        write_submission(out, answers)
        return 0
    except LexError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _questions_path(data_dir: Path) -> Path:
    primary = data_dir / "questions.csv"
    if primary.is_file():
        return primary
    demo = data_dir / "demo_questions.csv"
    if demo.is_file():
        return demo
    return primary


if __name__ == "__main__":
    raise SystemExit(main())
