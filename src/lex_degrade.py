"""Ступени деградации пайплайна по остатку времени."""

from __future__ import annotations

import sys
from enum import Enum


class PipelineStage(str, Enum):
    FULL = "full"
    NO_RERANK = "no_rerank"
    NO_CLASSIFIER = "no_classifier"
    TEMPLATE = "template"


# Секунды остатка на вопрос: ниже порога — следующая ступень.
NO_RERANK_BELOW_S = 20.0
NO_CLASSIFIER_BELOW_S = 8.0
TEMPLATE_BELOW_S = 2.0


def choose_stage(remaining_s: float) -> PipelineStage:
    if remaining_s < TEMPLATE_BELOW_S:
        return PipelineStage.TEMPLATE
    if remaining_s < NO_CLASSIFIER_BELOW_S:
        return PipelineStage.NO_CLASSIFIER
    if remaining_s < NO_RERANK_BELOW_S:
        return PipelineStage.NO_RERANK
    return PipelineStage.FULL


def log_stage(question_id: str, stage: PipelineStage, remaining_s: float) -> None:
    print(
        f"degrade q={question_id} stage={stage.value} remaining={remaining_s:.1f}s",
        file=sys.stderr,
    )
