from __future__ import annotations

import json
import math
import random
from pathlib import Path

import httpx
import pytest

from lex_deadline import Deadline
from lex_errors import ModelGatewayError
from lex_gateway import ProxyGateway
from lex_models import ChatTurn, FinishReason
from lex_retry import RetryPolicy, WallClock
from lex_settings import Settings


class FakeSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)


def _settings() -> Settings:
    return Settings.from_environ(
        {
            "LLM_BASE_URL": "http://proxy.test/v1",
            "LLM_API_KEY": "secret-key",
            "LLM_MODEL": "deepseek",
            "EMB_MODEL": "bge-m3",
            "RERANKER_MODEL": "qwen-rerank",
            "MAX_WORKERS": "2",
        }
    )


def _gateway(handler, sleeper: FakeSleeper | None = None) -> ProxyGateway:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="http://proxy.test/v1", transport=transport)
    policy = RetryPolicy.standard(rng=random.Random(0))
    return ProxyGateway(
        _settings(),
        client=client,
        policy=policy,
        clock=WallClock(),
        sleeper=sleeper or FakeSleeper(),
    )


def test_complete_reads_content_and_sends_thinking_disabled() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["path"] = request.url.path
        seen["thinking"] = body.get("thinking")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "срок 30 дней",
                            "reasoning_content": "не читать",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    gateway = _gateway(handler)
    result = gateway.complete([ChatTurn(role="user", content="срок?")])
    assert result.content == "срок 30 дней"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["thinking"] == {"type": "disabled"}


def test_empty_content_retries_then_fails() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
        )

    sleeper = FakeSleeper()
    gateway = _gateway(handler, sleeper)
    with pytest.raises(ModelGatewayError, match="пустой"):
        gateway.complete([ChatTurn(role="user", content="?")])
    assert hits["n"] == 4
    assert sleeper.delays


def _chat_reply(content: object, finish_reason: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": content}, "finish_reason": finish_reason}
            ]
        },
    )


def test_json_mode_is_requested_only_by_complete_json() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return _chat_reply('{"a": 1}', "stop")

    gateway = _gateway(handler)
    gateway.complete([ChatTurn(role="user", content="?")])
    result = gateway.complete_json([ChatTurn(role="user", content="json?")])
    assert "response_format" not in bodies[0]
    assert bodies[1]["response_format"] == {"type": "json_object"}
    assert bodies[1]["thinking"] == {"type": "disabled"}
    assert result.content == '{"a": 1}'
    assert result.finish is FinishReason.STOP


@pytest.mark.parametrize(
    ("content", "reported", "finish"),
    [
        ("   ", "stop", FinishReason.STOP),
        (None, "stop", FinishReason.STOP),
        ('{"claims": [', "length", FinishReason.LENGTH),
        ("", "content_filter", FinishReason.CONTENT_FILTER),
        ("{}", "tool_calls", FinishReason.UNRECOGNIZED),
        ("{}", None, FinishReason.UNRECOGNIZED),
    ],
)
def test_complete_json_reports_finish_reason_without_retrying(
    content: str | None, reported: str | None, finish: FinishReason
) -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return _chat_reply(content, reported)

    result = _gateway(handler).complete_json([ChatTurn(role="user", content="?")])
    assert result.finish is finish
    assert result.content == (content or "")
    assert hits["n"] == 1


@pytest.mark.parametrize("reported", ["insufficient_system_resource", "aborted"])
def test_complete_json_repeats_request_interrupted_by_provider(reported: str) -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return _chat_reply("", reported)
        return _chat_reply("{}", "stop")

    result = _gateway(handler).complete_json([ChatTurn(role="user", content="?")])
    assert result.content == "{}"
    assert hits["n"] == 2


def test_complete_json_gives_up_when_provider_keeps_interrupting() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return _chat_reply("", "aborted")

    with pytest.raises(ModelGatewayError, match="прервана"):
        _gateway(handler).complete_json([ChatTurn(role="user", content="?")])
    assert hits["n"] == 4


def test_complete_json_rejects_reply_without_choices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ModelGatewayError, match="некорректный ответ"):
        _gateway(handler).complete_json([ChatTurn(role="user", content="?")])


def test_error_message_does_not_repeat_key_or_response_body() -> None:
    key = _settings().llm_api_key.get()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"invalid api key {key}"})

    with pytest.raises(ModelGatewayError) as raised:
        _gateway(handler).complete_json([ChatTurn(role="user", content="?")])
    assert str(raised.value) == "прокси HTTP 401"
    assert key not in repr(raised.value)


def test_retry_after_429() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1.5"}, json={})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    sleeper = FakeSleeper()
    gateway = _gateway(handler, sleeper)
    result = gateway.complete([ChatTurn(role="user", content="?")])
    assert result.content == "ok"
    assert sleeper.delays[0] >= 1.5


def test_http_400_does_not_retry() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(400, json={"error": "bad"})

    gateway = _gateway(handler)
    with pytest.raises(ModelGatewayError, match="HTTP 400"):
        gateway.complete([ChatTurn(role="user", content="?")])
    assert hits["n"] == 1


def test_embeddings_are_l2_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]},
        )

    gateway = _gateway(handler)
    vectors = gateway.embed(["текст"])
    assert len(vectors) == 1
    assert math.isclose(math.sqrt(sum(x * x for x in vectors[0])), 1.0)


def test_rerank_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["query"] == "аренда"
        assert body["documents"] == ["a", "b"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    gateway = _gateway(handler)
    ranked = gateway.rerank("аренда", ["a", "b"])
    assert ranked[0].index == 1
    assert ranked[0].score == 0.9


def test_archive_includes_gateway_modules(tmp_path: Path) -> None:
    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    try:
        from build_archive import SRC_MODULES, ArchiveManifest, build
    finally:
        sys.path.pop(0)
    dest = tmp_path / "s.zip"
    build(ArchiveManifest.default(repo), dest)
    assert "lex_gateway.py" in SRC_MODULES
    import zipfile

    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
    assert "lex_gateway.py" in names
    assert "lex_models.py" in names
    assert "lex_retry.py" in names
    assert "lex_deadline.py" in names


class _StuckClock:
    def monotonic(self) -> float:
        return 100.0


def test_expired_deadline_skips_http() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="http://proxy.test/v1", transport=transport)
    clock = _StuckClock()
    deadline = Deadline(end_monotonic=1.0, reserve_s=0.0)
    gateway = ProxyGateway(
        _settings(),
        client=client,
        policy=RetryPolicy.standard(rng=random.Random(0), deadline_monotonic=1.0),
        clock=clock,
        sleeper=FakeSleeper(),
        deadline=deadline,
    )
    with pytest.raises(ModelGatewayError, match="таймаут"):
        gateway.complete([ChatTurn(role="user", content="?")])
    assert hits["n"] == 0
