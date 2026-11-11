"""Чтение questions.csv и атомарная запись submission.csv."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from lex_errors import SubmissionError

QUESTION_COLUMNS = ("id", "question")
ANSWER_COLUMNS = ("id", "answer")
MAX_ANSWER_LENGTH = 10_000
NO_PROVISIONS_YET = (
    "В базе не найдено положений, подтверждающих ответ. По памяти не отвечаем."
)


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    text: str


def load_questions(path: Path) -> list[Question]:
    if not path.is_file():
        raise SubmissionError(f"нет файла вопросов: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SubmissionError("questions.csv без заголовка")
        fields = [name.strip() for name in reader.fieldnames]
        missing = set(QUESTION_COLUMNS) - set(fields)
        if missing:
            raise SubmissionError(
                f"ожидались колонки {QUESTION_COLUMNS}, получены {tuple(fields)}"
            )
        rows: list[Question] = []
        seen: set[str] = set()
        for raw in reader:
            qid = (raw.get("id") or "").strip()
            text = raw.get("question") or ""
            if not qid:
                raise SubmissionError("пустой id в questions.csv")
            if qid in seen:
                raise SubmissionError(f"повтор id: {qid}")
            seen.add(qid)
            rows.append(Question(id=qid, text=text))
    if not rows:
        raise SubmissionError("questions.csv пуст")
    return rows


def placeholder_answer(_question: Question) -> str:
    return NO_PROVISIONS_YET


def write_submission(path: Path, answers: dict[str, str]) -> None:
    _validate(answers)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANSWER_COLUMNS))
        writer.writeheader()
        for qid, answer in answers.items():
            writer.writerow({"id": qid, "answer": answer})
    os.replace(tmp, path)


def _validate(answers: dict[str, str]) -> None:
    if not answers:
        raise SubmissionError("нет ответов для записи")
    for qid, answer in answers.items():
        if not qid:
            raise SubmissionError("пустой id в ответах")
        if not answer or not answer.strip():
            raise SubmissionError(f"пустой ответ для {qid}")
        if len(answer) > MAX_ANSWER_LENGTH:
            raise SubmissionError(f"ответ {qid} длиннее {MAX_ANSWER_LENGTH} символов")
