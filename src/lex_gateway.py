"""Адаптер OpenAI-совместимого прокси на httpx."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import httpx

from lex_errors import ModelGatewayError
from lex_models import ChatResult, ChatTurn, RankedDocument
from lex_retry import (
    RETRY_STATUSES,
    Clock,
    RetryPolicy,
    Sleeper,
    WallClock,
    WallSleeper,
    parse_retry_after,
    run_with_retries,
)
from lex_secrets import Secret
from lex_settings import Settings

T = TypeVar("T")

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 90.0
THINKING_DISABLED: dict[str, Any] = {"type": "disabled"}


class ProxyGateway:
    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        policy: RetryPolicy | None = None,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        limits = httpx.Limits(
            max_connections=max(4, settings.max_workers * 2),
            max_keepalive_connections=max(2, settings.max_workers),
        )
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=READ_TIMEOUT,
            write=30.0,
            pool=CONNECT_TIMEOUT,
        )
        self._client = client or httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            headers=_auth_headers(settings.llm_api_key),
            timeout=timeout,
            limits=limits,
        )
        self._policy = policy or RetryPolicy.standard()
        self._clock = clock or WallClock()
        self._sleeper = sleeper or WallSleeper()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(self, turns: Sequence[ChatTurn]) -> ChatResult:
        payload = {
            "model": self._settings.llm_model,
            "messages": [{"role": t.role, "content": t.content} for t in turns],
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "thinking": THINKING_DISABLED,
        }

        def parse() -> ChatResult:
            data = self._send("/chat/completions", payload)
            try:
                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                finish = str(choice.get("finish_reason") or "")
            except (KeyError, IndexError, TypeError) as exc:
                raise ModelGatewayError("некорректный ответ chat/completions") from exc
            if finish == "length" or not str(content).strip():
                raise ModelGatewayError(
                    f"пустой content или обрезка (finish_reason={finish!r})"
                )
            return ChatResult(content=str(content), finish_reason=finish)

        return self._call(parse)

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        payload = {
            "model": self._settings.embedding_model,
            "input": list(texts),
        }

        def parse() -> tuple[tuple[float, ...], ...]:
            data = self._send("/embeddings", payload)
            try:
                rows = sorted(data["data"], key=lambda item: int(item["index"]))
                vectors = tuple(_l2_normalize(row["embedding"]) for row in rows)
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelGatewayError("некорректный ответ embeddings") from exc
            if len(vectors) != len(texts):
                raise ModelGatewayError("число векторов не совпало с входом")
            return vectors

        return self._call(parse)

    def rerank(
        self, query: str, documents: Sequence[str]
    ) -> tuple[RankedDocument, ...]:
        payload = {
            "model": self._settings.reranker_model,
            "query": query,
            "documents": list(documents),
        }

        def parse() -> tuple[RankedDocument, ...]:
            data = self._send("/rerank", payload)
            try:
                results = data["results"]
                ranked = tuple(
                    RankedDocument(
                        index=int(item["index"]),
                        score=float(item["relevance_score"]),
                    )
                    for item in results
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelGatewayError("некорректный ответ rerank") from exc
            return ranked

        return self._call(parse)

    def _call(self, attempt: Callable[[], T]) -> T:
        try:
            return run_with_retries(
                self._policy,
                self._clock,
                self._sleeper,
                attempt,
                should_retry=_should_retry,
            )
        except _RetryableHttpError as exc:
            raise ModelGatewayError(f"прокси HTTP {exc.status}") from exc

    def _send(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ModelGatewayError("таймаут прокси") from exc
        except httpx.RequestError as exc:
            raise ModelGatewayError(f"сеть: {exc.__class__.__name__}") from exc
        if response.status_code in RETRY_STATUSES:
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            raise _RetryableHttpError(response.status_code, retry_after)
        if response.status_code >= 400:
            raise ModelGatewayError(f"прокси HTTP {response.status_code}")
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ModelGatewayError("тело ответа не JSON") from exc
        if not isinstance(data, dict):
            raise ModelGatewayError("тело ответа не объект")
        return data


class _RetryableHttpError(Exception):
    def __init__(self, status: int, retry_after: float | None) -> None:
        super().__init__(status)
        self.status = status
        self.retry_after = retry_after


def _should_retry(exc: Exception) -> tuple[bool, float | None]:
    if isinstance(exc, _RetryableHttpError):
        return True, exc.retry_after
    if isinstance(exc, ModelGatewayError):
        cause = exc.__cause__
        if isinstance(cause, (httpx.TimeoutException, httpx.RequestError)):
            return True, None
        if "пустой content" in str(exc) or "обрезка" in str(exc):
            return True, None
    return False, None


def _auth_headers(key: Secret) -> dict[str, str]:
    token = key.get()
    if not token:
        return {"Accept": "application/json"}
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    squares = sum(float(x) * float(x) for x in values)
    if squares <= 0:
        return tuple(float(x) for x in values)
    scale = 1.0 / math.sqrt(squares)
    return tuple(float(x) * scale for x in values)
