"""Shared fixtures: no test may see the developer's real endpoint config,
and HTTP tests talk to a local fake OpenAI-compatible server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

LLM_ENV = [f"{app}_LLM_{s}" for app in ("ADJUDICATE", "STYLEFIX", "PROOFIX")
           for s in ("URL", "MODEL", "KEY")]


@pytest.fixture(autouse=True)
def isolated_llm_config(tmp_path_factory, monkeypatch):
    """An empty XDG config home, no *_LLM_* variables, and an Ollama address
    that refuses connections at once."""
    xdg = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    return xdg


class FakeServer:
    def __init__(self):
        self.models = ["served-model"]   # listed by /v1/models and /api/tags
        self.models_status = 200         # status for GET /v1/models
        self.reject = set()              # chat body fields answered with HTTP 400
        self.requests = []               # every chat body received, parsed
        self.auth = []                   # Authorization header of every request


@pytest.fixture
def server():
    state = FakeServer()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state.auth.append(self.headers.get("Authorization"))
            if self.path == "/v1/models":
                if state.models_status != 200:
                    return self._send(state.models_status, {"error": "down"})
                return self._send(200, {"data": [{"id": m} for m in state.models]})
            if self.path == "/api/tags":           # Ollama's own listing
                return self._send(200, {"models": [{"name": m} for m in state.models]})
            self._send(404, {})

        def do_POST(self):
            state.auth.append(self.headers.get("Authorization"))
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            bad = state.reject & body.keys()
            if bad:
                return self._send(400, {"error": f"unrecognized fields: {sorted(bad)}"})
            self._send(200, {"choices": [{"message": {"content": '{"decisions": []}'}}]})

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.base = f"http://127.0.0.1:{httpd.server_port}"
    state.url = state.base + "/v1"
    yield state
    httpd.shutdown()
    httpd.server_close()
