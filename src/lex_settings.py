"""Конфигурация процесса. Собирается на входе и передаётся явно."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lex_deadline import DEFAULT_RESERVE_S, DEFAULT_RUN_TIMEOUT_S
from lex_secrets import Secret

_DEFAULT_DATA_DIR = "data"


@dataclass(frozen=True, slots=True)
class SanitizerThresholds:
    """Пороги санитайзера базы. Подобраны на 21 документе базы и синтетике."""

    # Столько детекторов со слабым сигналом дают карантин и без сильного сигнала.
    weak_detectors_for_quarantine: int = 2
    # Слов со смешением алфавитов в абзаце, начиная с которых это маскировка.
    mixed_script_words_for_strong: int = 3
    # Сколько соседних абзацев читается вместе в поиске разбитой вставки.
    window_size: int = 3


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    llm_base_url: str
    llm_api_key: Secret
    llm_model: str
    embedding_model: str
    reranker_model: str
    temperature: float
    max_tokens: int
    max_workers: int
    top_k: int
    reranker_top_n: int
    run_timeout_s: float
    reserve_s: float
    sanitizer: SanitizerThresholds

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> Settings:
        data_dir = Path(environ.get("DATA_DIR", _DEFAULT_DATA_DIR))
        return cls(
            data_dir=data_dir,
            llm_base_url=_first(environ, ("LLM_BASE_URL", "OPENAI_BASE_URL"), ""),
            llm_api_key=Secret(_first(environ, ("LLM_API_KEY", "OPENAI_API_KEY"), "")),
            llm_model=_first(environ, ("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL"), ""),
            embedding_model=environ.get("EMB_MODEL", ""),
            reranker_model=environ.get("RERANKER_MODEL", ""),
            temperature=_float(environ.get("TEMPERATURE"), 0.0),
            max_tokens=_int(environ.get("MAX_TOKENS"), 1024),
            max_workers=_int(environ.get("MAX_WORKERS"), 4),
            top_k=_int(environ.get("TOP_K"), 8),
            reranker_top_n=_int(environ.get("RERANKER_TOP_N"), 8),
            run_timeout_s=_float(
                environ.get("RUN_TIMEOUT"), float(DEFAULT_RUN_TIMEOUT_S)
            ),
            reserve_s=_float(environ.get("RESERVE_S"), DEFAULT_RESERVE_S),
            sanitizer=SanitizerThresholds(),
        )


def _first(environ: Mapping[str, str], keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        value = environ.get(key)
        if value:
            return value
    return default


def _int(raw: str | None, default: int) -> int:
    if raw is None or raw == "":
        return default
    return int(raw)


def _float(raw: str | None, default: float) -> float:
    if raw is None or raw == "":
        return default
    return float(raw)
