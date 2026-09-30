"""Minimal OpenAI-compatible chat client, and endpoint resolution from flags, environment and config files."""

from __future__ import annotations

import json
import os
import re
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


class LLMError(RuntimeError):
    pass


class ConfigError(LLMError):
    """An invalid endpoint setting; the message names the file or variable."""


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
    sources: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

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
        for name in _OPTIONAL_FIELDS:
            if name in self._dropped:
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


def ollama_endpoint() -> Endpoint | None:
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=2) as resp:
            models = [m["name"] for m in json.load(resp).get("models", [])]
    except Exception:
        return None
    return Endpoint(url=url + "/v1", model=models[0] if models else "", name="ollama")


# [llm] keys and the types they accept.
_CONFIG_KEYS: dict[str, tuple[type, ...]] = {
    "url": (str,), "model": (str,), "api_key": (str,), "api_key_env": (str,),
    "timeout": (int, float), "thinking_switch": (bool,),
}
# Settings that belong to one server: a layer that names a url drops these
# from the layers below it.
_SERVER_KEYS = ("model", "api_key", "api_key_env", "thinking_switch")
# The two ways to give a key: a layer that sets either drops both below it.
_KEY_FORMS = ("api_key", "api_key_env")


def config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def read_config(path: Path) -> dict:
    """The validated [llm] table of a config file as {key: (value, source)};
    {} if the file does not exist. api_key_env is kept unresolved, so a
    higher layer's url or key can discard it before its variable is read."""
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"{path}: cannot read: {e}") from e
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: not valid TOML: {e}") from e
    table = data.get("llm", {})
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: 'llm' must be a table")
    for k, v in table.items():
        if k not in _CONFIG_KEYS:
            raise ConfigError(f"{path}: unknown key '{k}' in [llm] "
                              f"(known: {', '.join(_CONFIG_KEYS)})")
        types = _CONFIG_KEYS[k]
        if not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
            raise ConfigError(f"{path}: [llm] {k} must be "
                              f"{' or '.join(t.__name__ for t in types)}, not {v!r}")
        if isinstance(v, str) and not v.strip():
            raise ConfigError(f"{path}: [llm] {k} is empty; remove the line to leave it unset")
        if k == "timeout" and v <= 0:
            raise ConfigError(f"{path}: [llm] timeout must be a positive number of seconds, not {v!r}")
    if "api_key" in table and "api_key_env" in table:
        raise ConfigError(f"{path}: set api_key or api_key_env in [llm], not both")
    return {k: (v, str(path)) for k, v in table.items()}


def env_layer(prefix: str) -> dict:
    """Settings from <prefix>_LLM_URL, _MODEL and _KEY; empty values are unset."""
    out = {}
    for key, suffix in (("url", "URL"), ("model", "MODEL"), ("api_key", "KEY")):
        name = f"{prefix}_LLM_{suffix}"
        if os.environ.get(name):
            out[key] = (os.environ[name], name)
    return out


def merge_layers(layers: list[dict]) -> dict:
    """Merge {key: (value, source)} layers field by field, highest first."""
    out: dict = {}
    for layer in reversed(layers):
        if "url" in layer:
            for k in _SERVER_KEYS:
                out.pop(k, None)
        if any(k in layer for k in _KEY_FORMS):
            for k in _KEY_FORMS:
                out.pop(k, None)
        out.update(layer)
    return out


def _resolve_key_env(settings: dict) -> dict:
    """settings with a surviving api_key_env replaced by the key it names."""
    if "api_key_env" not in settings:
        return settings
    settings = dict(settings)
    var, src = settings.pop("api_key_env")
    if not os.environ.get(var):
        raise ConfigError(f"{src}: api_key_env names {var}, which is not set")
    settings["api_key"] = (os.environ[var], f"{src} (api_key_env {var})")
    return settings


def list_models_detail(ep: Endpoint, timeout: float = 5.0) -> tuple[list[str] | None, str]:
    """(every model id the server lists at /models, "") or, if that fails,
    (None, why): the HTTP status and the start of the body, or the
    connection error."""
    req = urllib.request.Request(ep.url.rstrip("/") + "/models", headers=ep._headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode(errors="replace").strip()
        except Exception:
            body = ""
        body = " ".join(body.split())[:200]
        return None, f"HTTP {e.code}" + (f": {body}" if body else "")
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except ValueError as e:
        return None, f"not a JSON response ({e})"
    except Exception as e:
        return None, str(e) or type(e).__name__
    items = data.get("data", []) if isinstance(data, dict) else []
    # A gateway routing any name (LiteLLM lists "*", "openai/*") offers a
    # pattern, not a model that can be requested.
    return [m["id"] for m in items
            if isinstance(m, dict) and m.get("id") and "*" not in m["id"]], ""


def list_models(ep: Endpoint, timeout: float = 5.0) -> list[str] | None:
    """Every model id the server lists at /models, or None if that fails."""
    return list_models_detail(ep, timeout)[0]


def discover_model(ep: Endpoint, timeout: float = 5.0) -> str | None:
    models = list_models(ep, timeout)
    return models[0] if models else None


_FLAG_SOURCE = {"url": "--url", "model": "--model", "api_key": "key argument",
                "timeout": "--timeout"}


def resolve_endpoint(app: str, preset: str = "auto", url: str | None = None,
                     model: str | None = None, key: str | None = None,
                     think: bool = False, timeout: float | None = None) -> Endpoint | None:
    """The endpoint `app` should use, or None for no model.

    preset: auto | ollama | none. For auto, settings come from, highest
    first: the arguments, <APP>_LLM_* variables, ~/.config/<app>/config.toml,
    ADJUDICATE_LLM_* variables, ~/.config/adjudicate/config.toml; with no
    url anywhere, a local Ollama is used if one is running with a model.
    ollama means local Ollama with only the arguments applied. The result's
    `sources` says where each setting came from. Raises ConfigError for an
    invalid setting or an endpoint with no usable model.
    """
    if preset == "none":
        return None
    if preset not in ("auto", "ollama"):
        raise ConfigError(f"unknown endpoint preset '{preset}' (choose auto, ollama or none)")
    if timeout is not None and timeout <= 0:
        raise ConfigError(f"--timeout must be a positive number of seconds, not {timeout!r}")
    # Empty strings leave a setting unset, as empty variables do; a timeout is
    # set whenever it is given.
    flags = {k: (v, _FLAG_SOURCE[k]) for k, v in
             (("url", url), ("model", model), ("api_key", key)) if v}
    if timeout is not None:
        flags["timeout"] = (timeout, _FLAG_SOURCE["timeout"])
    if preset == "ollama":
        settings = flags
    else:
        home = config_home()
        settings = merge_layers([
            flags,
            env_layer(app.upper()),
            read_config(home / app / "config.toml"),
            env_layer("ADJUDICATE"),
            read_config(home / "adjudicate" / "config.toml"),
        ])
    settings = _resolve_key_env(settings)
    values = {k: v for k, (v, _) in settings.items()}
    sources = {"timeout": "default", "thinking_switch": "default",
               **{k: s for k, (_, s) in settings.items()}}
    t = float(values.get("timeout", DEFAULT_TIMEOUT))
    if "url" in values:
        ep = Endpoint(url=values["url"], model=values.get("model", ""),
                      key=values.get("api_key", ""), think=think, timeout=t,
                      thinking_switch=values.get("thinking_switch", False))
    else:
        ep = ollama_endpoint()
        if ep is None:
            return None
        sources["url"] = "local Ollama"
        if not values.get("model") and ep.model:
            sources["model"] = "local Ollama"
        ep.model = values.get("model") or ep.model
        ep.key = values.get("api_key", "")
        ep.think, ep.timeout = think, t
        ep.thinking_switch = values.get("thinking_switch", False)
    if not ep.model:
        ep.model = discover_model(ep) or ""
        sources["model"] = "listed by the server"
    if not ep.model:
        if ep.name == "ollama" and preset == "auto":
            return None          # Ollama is running but has no model pulled
        raise ConfigError(f"no model set for {ep.url} and the server lists none; "
                          "set `model` in config.toml or pass --model")
    ep.sources = sources
    return ep
