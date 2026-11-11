"""Локальный OpenAI-совместимый прокси: детерминированные ответы и сбои."""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class ProxyScript:
    """Сценарий ответа. fail_times: сколько первых запросов отдать status."""

    fail_status: int = 429
    fail_times: int = 0
    delay_s: float = 0.0
    empty_content: bool = False


@dataclass
class ProxyState:
    script: ProxyScript
    hits: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class FakeOpenAIServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        script: ProxyScript | None = None,
    ) -> None:
        self._state = ProxyState(script=script or ProxyScript())
        handler = _make_handler(self._state)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> str:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _make_handler(state: ProxyState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "not json"})
                return
            if not isinstance(payload, dict):
                self._json(400, {"error": "not object"})
                return
            script = _script_from_query(self.path, state.script)
            with state.lock:
                state.hits += 1
                hit = state.hits
            if script.delay_s > 0:
                time.sleep(script.delay_s)
            if hit <= script.fail_times:
                self._json(script.fail_status, {"error": "injected"})
                return
            path = urlparse(self.path).path.rstrip("/")
            if path.endswith("/chat/completions"):
                self._json(200, _chat_body(payload, script))
                return
            if path.endswith("/embeddings"):
                self._json(200, _embed_body(payload))
                return
            if path.endswith("/rerank"):
                self._json(200, _rerank_body(payload))
                return
            self._json(404, {"error": "unknown path"})

        def _json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if status == 429:
                self.send_header("Retry-After", "0.01")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _script_from_query(path: str, default: ProxyScript) -> ProxyScript:
    query = parse_qs(urlparse(path).query)
    fail_times = default.fail_times
    if "fail_times" in query:
        fail_times = int(query["fail_times"][0])
    fail_status = default.fail_status
    if "fail_status" in query:
        fail_status = int(query["fail_status"][0])
    delay_s = default.delay_s
    if "delay" in query:
        delay_s = float(query["delay"][0])
    empty = default.empty_content or query.get("empty", ["0"])[0] == "1"
    return ProxyScript(
        fail_status=fail_status,
        fail_times=fail_times,
        delay_s=delay_s,
        empty_content=empty,
    )


def _chat_body(payload: dict[str, Any], script: ProxyScript) -> dict[str, Any]:
    content = "" if script.empty_content else "ok"
    finish = "length" if script.empty_content else "stop"
    _ = payload
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": "ignore"},
                "finish_reason": finish,
            }
        ]
    }


def _embed_body(payload: dict[str, Any]) -> dict[str, Any]:
    texts = payload.get("input") or []
    if isinstance(texts, str):
        texts = [texts]
    data = []
    for index, _text in enumerate(texts):
        data.append({"index": index, "embedding": [1.0, 0.0, 0.0]})
    return {"data": data}


def _rerank_body(payload: dict[str, Any]) -> dict[str, Any]:
    documents = payload.get("documents") or []
    results = [
        {"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(len(documents))
    ]
    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Фейковый OpenAI-прокси")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fail-times", type=int, default=0)
    parser.add_argument("--fail-status", type=int, default=429)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    script = ProxyScript(
        fail_status=args.fail_status,
        fail_times=args.fail_times,
        delay_s=args.delay,
    )
    server = FakeOpenAIServer(host=args.host, port=args.port, script=script)
    url = server.start()
    print(url, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
