"""Minimal OpenAI-compatible chat client plus endpoint presets."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


class LLMError(RuntimeError):
    pass


DEFAULT_TIMEOUT = 120.0

# Request fields a strict server may reject with 400/422, in the order they
# are dropped on retry.
_OPTIONAL_FIELDS = ("chat_template_kwargs", "response_format")


@dataclass
class Endpoint:
    url: str            # base URL ending in /v1
    model: str
    key: str = ""
    name: str = "custom"
    think: bool = False   # let reasoning models think before answering
    timeout: float = DEFAULT_TIMEOUT
    thinking_switch: bool = False   # server honours chat_template_kwargs.enable_thinking
    _dropped: set[str] = field(default_factory=set, init=False, repr=False, compare=False)

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.key or 'none'}"}

    def chat(self, system: str, user: str, timeout: float | None = None,
             json_mode: bool = True) -> str:
        timeout = self.timeout if timeout is None else timeout
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if self.think or self.thinking_switch:
            # Reasoning models spend tokens thinking; this task doesn't need it.
            body["chat_template_kwargs"] = {"enable_thinking": self.think}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for name in self._dropped:
            body.pop(name, None)
        req = urllib.request.Request(
            self.url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (400, 422):
                # A strict server rejecting an extension: drop it for the rest of the run.
                for name in _OPTIONAL_FIELDS:
                    if name in body:
                        self._dropped.add(name)
                        return self.chat(system, user, timeout, json_mode)
            raise LLMError(f"{self.name} returned HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"cannot reach {self.name} at {self.url}: {e}") from e
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape from {self.name}: {str(data)[:300]}") from e
        return strip_thinking(content)

    def reachable(self, timeout: float = 3.0) -> bool:
        req = urllib.request.Request(
            self.url.rstrip("/") + "/models",
            headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:
            return False


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def dgx_endpoint() -> Endpoint | None:
    """Endpoint for the dgx-llm LiteLLM gateway, read from its .env."""
    env_dir = Path(os.environ.get("DGX_LLM_DIR", Path.home() / "Code" / "dgx-llm"))
    env = read_env_file(env_dir / ".env")
    host = env.get("DGX_HTTP_HOST")
    if not host:
        return None
    return Endpoint(url=f"http://{host}:4000/v1",
                    model=env.get("PRIMARY_MODEL_ALIAS", ""),
                    key=env.get("LITELLM_MASTER_KEY", ""), name="dgx")


def ollama_endpoint() -> Endpoint | None:
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=2) as resp:
            models = [m["name"] for m in json.load(resp).get("models", [])]
    except Exception:
        return None
    return Endpoint(url=url + "/v1", model=models[0] if models else "", name="ollama")


def resolve_endpoint(preset: str, model: str | None = None, url: str | None = None,
                     key: str | None = None, think: bool = False,
                     timeout: float = 120.0) -> Endpoint | None:
    """Pick an endpoint. Explicit args and STYLEFIX_LLM_* env vars win over presets.

    preset: auto | dgx | ollama | none
    """
    url = url or os.environ.get("STYLEFIX_LLM_URL")
    model = model or os.environ.get("STYLEFIX_LLM_MODEL")
    key = key or os.environ.get("STYLEFIX_LLM_KEY")
    if preset == "none":
        return None
    if url:
        return Endpoint(url=url, model=model or "", key=key or "", name="custom",
                        think=think, timeout=timeout)
    candidates = {"dgx": [dgx_endpoint], "ollama": [ollama_endpoint],
                  "auto": [dgx_endpoint, ollama_endpoint]}[preset]
    for make in candidates:
        ep = make()
        if ep is None:
            continue
        if preset == "auto" and not ep.reachable():
            continue
        if model:
            ep.model = model
        if key:
            ep.key = key
        ep.think = think
        ep.timeout = timeout
        return ep
    return None
