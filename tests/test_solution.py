from __future__ import annotations

from pathlib import Path

from lex_deadline import Deadline
from lex_degrade import PipelineStage
from lex_errors import ModelGatewayError
from lex_submission import TIME_LIMIT_STUB, Question
from solution import main


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _write_questions(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "questions.csv").write_text(
        "id,question\nq1,срок?\nq2,ндс?\nq3,аренда?\n",
        encoding="utf-8",
    )


def test_expired_deadline_fills_stub(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_questions(data)
    clock = FakeClock(0.0)
    calls: list[str] = []

    def answer(question: Question, stage: PipelineStage, _d: Deadline) -> str:
        calls.append(question.id)
        clock.advance(100.0)
        return f"ответ {question.id} {stage.value}"

    code = main(
        {"DATA_DIR": str(data), "RUN_TIMEOUT": "5", "RESERVE_S": "0"},
        tmp_path,
        clock=clock,
        answer_fn=answer,
        install_signals=False,
    )
    assert code == 0
    text = (tmp_path / "submission.csv").read_text(encoding="utf-8")
    assert "q1" in text
    assert TIME_LIMIT_STUB in text
    assert calls == ["q1"]


def test_gateway_error_becomes_stub(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_questions(data)

    def answer(_q: Question, _s: PipelineStage, _d: Deadline) -> str:
        raise ModelGatewayError("таймаут прокси")

    code = main(
        {"DATA_DIR": str(data), "RUN_TIMEOUT": "1200", "RESERVE_S": "0"},
        tmp_path,
        clock=FakeClock(),
        answer_fn=answer,
        install_signals=False,
    )
    assert code == 0
    text = (tmp_path / "submission.csv").read_text(encoding="utf-8")
    assert text.count(TIME_LIMIT_STUB) == 3


def test_sigterm_flushes_complete_csv(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_questions(data)
    from lex_deadline import Deadline as Dl
    from lex_submission import load_questions
    from solution import _Run

    questions = load_questions(data / "questions.csv")
    clock = FakeClock()
    deadline = Dl.from_now(100.0, clock=clock, reserve_s=0.0)
    run = _Run(
        questions=questions,
        out=tmp_path / "submission.csv",
        deadline=deadline,
        clock=clock,
        answer_fn=lambda q, _s, _d: f"ok {q.id}",
    )
    run.answers["q1"] = "ok q1"
    run.request_stop()
    rows = (tmp_path / "submission.csv").read_text(encoding="utf-8")
    assert "ok q1" in rows
    assert TIME_LIMIT_STUB in rows
    assert "q2" in rows
    assert "q3" in rows
    assert run.stop
