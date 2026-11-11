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

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("", "/", "/index.html"):
                with state.lock:
                    hits = state.hits
                page = _INDEX_HTML.replace("{{HITS}}", str(hits))
                self._raw(200, "text/html; charset=utf-8", page.encode("utf-8"))
                return
            if path.rstrip("/") == "/health":
                with state.lock:
                    hits = state.hits
                self._json(200, {"ok": True, "hits": hits})
                return
            self._json(404, {"error": "unknown path"})

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            if path in ("", "/", "/index.html", "/health"):
                self.send_response(200)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

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
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            extra = {}
            if status == 429:
                extra["Retry-After"] = "0.01"
            self._raw(status, "application/json; charset=utf-8", data, extra)

        def _raw(
            self,
            status: int,
            content_type: str,
            data: bytes,
            extra: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if extra:
                for key, value in extra.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

    return Handler


_INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Фейковый прокси</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font: 16px/1.4 system-ui, sans-serif;
      max-width: 42rem;
      margin: 2rem auto;
      padding: 0 1rem;
    }
    code, pre { font-family: ui-monospace, Consolas, monospace; font-size: 0.9em; }
    textarea, button { font: inherit; }
    textarea { width: 100%; min-height: 6rem; }
    pre { white-space: pre-wrap; background: #1111; padding: 0.75rem; }
    label { display: block; margin: 0.75rem 0 0.25rem; }
  </style>
</head>
<body>
  <h1>Фейковый OpenAI-прокси</h1>
  <p>Это не чат ассистента. Браузер ходил GET на API, поэтому раньше
     был 501. POST: <code>/v1/chat/completions</code>,
     <code>/v1/embeddings</code>, <code>/v1/rerank</code>.
     Запросов со старта: {{HITS}}.</p>
  <label for="q">Сообщение</label>
  <textarea id="q">Какой срок оплаты?</textarea>
  <p><button type="button" id="go">Отправить в /v1/chat/completions</button></p>
  <pre id="out"></pre>
  <script>
    document.getElementById("go").onclick = async () => {
      const text = document.getElementById("q").value;
      const out = document.getElementById("out");
      out.textContent = "…";
      try {
        const res = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "deepseek",
            messages: [{ role: "user", content: text }],
          }),
        });
        out.textContent = JSON.stringify(await res.json(), null, 2);
      } catch (err) {
        out.textContent = String(err);
      }
    };
  </script>
</body>
</html>
"""


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
