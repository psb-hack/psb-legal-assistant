"""Демо-вопросы и структурные ответы на них, составленные вручную по эталонам.

Общие данные golden-тестов рендерера и промпта. Архива базы и каталога data/
в репозитории нет: тесты, которые берут данные отсюда, без архива пропускаются.
"""

from __future__ import annotations

import csv
import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from lex_documents import ProvisionId, cite
from lex_knowledge_base import KnowledgeBase, load_knowledge_base
from lex_sanitizer import AdmittedProvision

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "knowledge-base.zip"
_QUESTIONS = REPO / "data" / "demo_questions.csv"
_REFERENCE_ANSWERS = REPO / "data" / "demo_output.csv"
_GOLDEN = Path(__file__).resolve().parent / "golden" / "answer_rendering.yaml"
CASES: list[dict[str, Any]] = yaml.safe_load(_GOLDEN.read_text(encoding="utf-8"))


@functools.cache
def base() -> KnowledgeBase:
    return load_knowledge_base(ARCHIVE)


@functools.cache
def questions() -> dict[str, str]:
    return _column(_QUESTIONS, "question")


@functools.cache
def reference_answers() -> dict[str, str]:
    return _column(_REFERENCE_ANSWERS, "answer")


def context(ids: Sequence[ProvisionId]) -> list[AdmittedProvision]:
    """Документы, на которые ссылается ответ, целиком — как их дал бы поиск."""
    cited_documents = {one.split(".")[0] for one in ids}
    return [
        AdmittedProvision(
            id=document.provision_id(provision),
            heading=cite(document, provision),
            text=provision.text,
        )
        for document in base().documents
        if document.id in cited_documents
        for provision in document.provisions
    ]


def model_reply(case: dict[str, Any]) -> tuple[str, list[ProvisionId]]:
    """JSON, который вернула бы модель, и идентификаторы положений из него."""
    ids: list[ProvisionId] = []
    claims = []
    for claim in case["claims"]:
        cited = [_ids_by_reference()[source] for source in claim["sources"]]
        ids.extend(cited)
        claims.append({"ids": cited, "quote": claim["quote"], "text": claim["text"]})
    reply = {
        "scenario": case["scenario"],
        "claims": claims,
        "missing": case["missing"],
        "verdict": case["verdict"],
    }
    return json.dumps(reply, ensure_ascii=False), ids


@functools.cache
def _ids_by_reference() -> dict[str, ProvisionId]:
    return {
        cite(document, provision): document.provision_id(provision)
        for document in base().documents
        for provision in document.provisions
    }


def _column(path: Path, name: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {row["id"]: row[name] for row in csv.DictReader(file)}
