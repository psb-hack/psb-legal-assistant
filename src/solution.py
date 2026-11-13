"""Точка входа сабмита: дедлайн, периодическая запись, SIGTERM."""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from lex_deadline import Deadline
from lex_degrade import PipelineStage, choose_stage, log_stage
from lex_errors import LexError, SubmissionError
from lex_retry import Clock, WallClock
from lex_settings import Settings
from lex_submission import (
    TIME_LIMIT_STUB,
    Question,
    load_questions,
    placeholder_answer,
    write_submission,
)

AnswerFn = Callable[[Question, PipelineStage, Deadline], str]


def main(
    environ: dict[str, str] | None = None,
    cwd: Path | None = None,
    *,
    clock: Clock | None = None,
    answer_fn: AnswerFn | None = None,
    install_signals: bool = True,
) -> int:
    env = os.environ if environ is None else environ
    workdir = Path.cwd() if cwd is None else cwd
    tick = clock if clock is not None else WallClock()
    settings = Settings.from_environ(env)
    questions_path = _questions_path(settings.data_dir)
    out = workdir / "submission.csv"
    try:
        questions = load_questions(questions_path)
    except LexError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    deadline = Deadline.from_now(
        settings.run_timeout_s, clock=tick, reserve_s=settings.reserve_s
    )
    run = _Run(
        questions=questions,
        out=out,
        deadline=deadline,
        clock=tick,
        answer_fn=answer_fn or _answer_stub,
    )
    if install_signals:
        _install_stop_signals(run)
    try:
        run.process()
        return 0
    except LexError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            run.flush()
        except SubmissionError:
            return 1
        return 1


class _Run:
    def __init__(
        self,
        *,
        questions: list[Question],
        out: Path,
        deadline: Deadline,
        clock: Clock,
        answer_fn: AnswerFn,
    ) -> None:
        self._questions = questions
        self._out = out
        self._deadline = deadline
        self._clock = clock
        self._answer_fn = answer_fn
        self.answers: dict[str, str] = {}
        self.stop = False

    def process(self) -> None:
        total = len(self._questions)
        for index, question in enumerate(self._questions):
            if self.stop or self._deadline.expired(self._clock):
                break
            left_n = total - index
            question_deadline = self._deadline.for_question(self._clock, left_n)
            remaining = question_deadline.remaining(self._clock)
            stage = choose_stage(remaining)
            log_stage(question.id, stage, remaining)
            try:
                text = self._answer_fn(question, stage, question_deadline)
            except LexError as exc:
                print(
                    f"q={question.id} {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                text = TIME_LIMIT_STUB
            if not text.strip():
                text = TIME_LIMIT_STUB
            self.answers[question.id] = text
            self.flush()
        self._fill_missing()
        self.flush()

    def flush(self) -> None:
        if not self._questions:
            return
        self._fill_missing()
        write_submission(self._out, dict(self.answers))

    def _fill_missing(self) -> None:
        for question in self._questions:
            if question.id not in self.answers:
                self.answers[question.id] = TIME_LIMIT_STUB

    def request_stop(self, *_args: object) -> None:
        self.stop = True
        try:
            self.flush()
        except SubmissionError as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)


def _answer_stub(
    question: Question, _stage: PipelineStage, _deadline: Deadline
) -> str:
    return placeholder_answer(question)


def _questions_path(data_dir: Path) -> Path:
    primary = data_dir / "questions.csv"
    if primary.is_file():
        return primary
    demo = data_dir / "demo_questions.csv"
    if demo.is_file():
        return demo
    return primary


def _install_stop_signals(run: _Run) -> None:
    signal.signal(signal.SIGTERM, run.request_stop)
    signal.signal(signal.SIGINT, run.request_stop)


if __name__ == "__main__":
    raise SystemExit(main())
