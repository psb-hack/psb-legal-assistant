from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx
import pytest

from lex_errors import SubmissionError
from lex_gateway import ProxyGateway
from lex_models import ChatTurn
from lex_settings import Settings
from lex_submission import NO_PROVISIONS_YET, check_submission, write_submission

REPO = Path(__file__).resolve().parents[1]


def _scripts() -> None:
    sys.path.insert(0, str(REPO / "scripts"))


def test_zip_hash_is_stable(tmp_path: Path) -> None:
    _scripts()
    from build_archive import ArchiveManifest, build

    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"
    build(ArchiveManifest.default(REPO), a)
    build(ArchiveManifest.default(REPO), b)
    assert (
        hashlib.sha256(a.read_bytes()).digest()
        == hashlib.sha256(b.read_bytes()).digest()
    )


def test_fake_proxy_chat_embed_rerank() -> None:
    _scripts()
    from fake_proxy import FakeOpenAIServer, ProxyScript

    server = FakeOpenAIServer(script=ProxyScript())
    base = server.start()
    try:
        settings = Settings.from_environ(
            {
                "LLM_BASE_URL": base,
                "LLM_API_KEY": "k",
                "LLM_MODEL": "deepseek",
                "EMB_MODEL": "bge-m3",
                "RERANKER_MODEL": "qwen",
                "MAX_WORKERS": "1",
            }
        )
        gateway = ProxyGateway(settings)
        try:
            chat = gateway.complete([ChatTurn(role="user", content="срок?")])
            assert chat.content == "ok"
            vectors = gateway.embed(["текст"])
            assert len(vectors) == 1
            ranked = gateway.rerank("q", ["a", "b"])
            assert ranked[0].index == 0
        finally:
            gateway.close()
    finally:
        server.stop()


def test_fake_proxy_429_then_ok() -> None:
    _scripts()
    from fake_proxy import FakeOpenAIServer, ProxyScript

    server = FakeOpenAIServer(script=ProxyScript(fail_status=429, fail_times=1))
    base = server.start()
    try:
        settings = Settings.from_environ(
            {
                "LLM_BASE_URL": base,
                "LLM_API_KEY": "k",
                "LLM_MODEL": "deepseek",
                "EMB_MODEL": "bge-m3",
                "RERANKER_MODEL": "qwen",
                "MAX_WORKERS": "1",
            }
        )
        gateway = ProxyGateway(settings)
        try:
            result = gateway.complete([ChatTurn(role="user", content="?")])
            assert result.content == "ok"
        finally:
            gateway.close()
    finally:
        server.stop()


def test_check_submission_same_as_write(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    write_submission(path, {"q1": NO_PROVISIONS_YET})
    loaded = check_submission(path, ["q1"])
    assert loaded["q1"] == NO_PROVISIONS_YET
    with pytest.raises(SubmissionError, match="не совпали"):
        check_submission(path, ["q2"])


def test_run_local_unpacks_zip_and_writes_submission(tmp_path: Path) -> None:
    _scripts()
    from build_archive import ArchiveManifest, build
    from run_platform import extract_archive, run_local

    archive = tmp_path / "s.zip"
    build(ArchiveManifest.default(REPO), archive)
    work = tmp_path / "run"
    extract_archive(archive, work)
    data = tmp_path / "data"
    data.mkdir()
    (data / "questions.csv").write_text("id,question\nq1,Срок?\n", encoding="utf-8")
    result = run_local(work, data, timeout_s=30)
    assert result.returncode == 0, result.stderr
    answers = check_submission(work / "submission.csv", ["q1"])
    assert answers["q1"].strip()


def test_docker_mode_skipped_without_daemon() -> None:
    _scripts()
    from run_platform import docker_available

    if docker_available():
        pytest.skip("проверка отсутствия docker не нужна, бинарь есть")
    assert docker_available() is False


def test_fake_proxy_get_index_is_html() -> None:
    _scripts()
    from fake_proxy import FakeOpenAIServer

    server = FakeOpenAIServer()
    base = server.start()
    try:
        root = "http://" + base.split("://", 1)[1].split("/", 1)[0]
        response = httpx.get(root + "/", timeout=5.0)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "chat/completions" in response.text
        health = httpx.get(root + "/health", timeout=5.0)
        assert health.json()["ok"] is True
    finally:
        server.stop()


def test_httpx_reaches_fake_proxy_health_shape() -> None:
    _scripts()
    from fake_proxy import FakeOpenAIServer

    server = FakeOpenAIServer()
    base = server.start()
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            json={"model": "x", "messages": []},
            timeout=5.0,
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ok"
    finally:
        server.stop()
