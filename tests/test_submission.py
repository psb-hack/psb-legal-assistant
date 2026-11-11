from pathlib import Path

import pytest

from lex_errors import SubmissionError
from lex_submission import (
    NO_PROVISIONS_YET,
    Question,
    check_submission,
    load_questions,
    placeholder_answer,
    write_submission,
)


def test_load_and_write_roundtrip(tmp_path: Path) -> None:
    questions = tmp_path / "questions.csv"
    questions.write_text(
        "id,question\n1,Сколько аренда?\n2,Что с НДС?\n", encoding="utf-8"
    )
    rows = load_questions(questions)
    assert [item.id for item in rows] == ["1", "2"]
    out = tmp_path / "submission.csv"
    write_submission(out, {item.id: placeholder_answer(item) for item in rows})
    text = out.read_text(encoding="utf-8")
    assert "id,answer" in text
    assert NO_PROVISIONS_YET in text


def test_duplicate_id_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    path.write_text("id,question\n1,a\n1,b\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="повтор"):
        load_questions(path)


def test_empty_answer_rejected(tmp_path: Path) -> None:
    with pytest.raises(SubmissionError, match="пустой ответ"):
        write_submission(tmp_path / "submission.csv", {"1": "  "})


def test_check_submission_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    with pytest.raises(SubmissionError, match="нет файла"):
        check_submission(path, ["1"])


def test_placeholder_has_no_demo_facts() -> None:
    text = placeholder_answer(Question(id="1", text="А-08/2024"))
    assert "А-08" not in text
    assert "450" not in text
