"""Interactive endpoint setup (--setup) and a non-interactive check
(--check-endpoint), shared by the tools built on adjudicate."""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import sys
import time
import tomllib
from pathlib import Path
from typing import TextIO

from .llm import (ConfigError, Endpoint, LLMError, config_home, list_models,
                  ollama_endpoint, resolve_endpoint)

ROWS = ("url", "model", "api_key", "timeout", "thinking_switch")
PING = 'Answer with the JSON object {"ok": true} and nothing else.'
_KEY_ORDER = ("url", "model", "api_key", "api_key_env", "timeout", "thinking_switch")
_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LLM_HEADER = re.compile(r"^[ \t]*\[llm\][ \t]*(?:#.*)?$", re.M)
_ANY_HEADER = re.compile(r"^[ \t]*\[", re.M)


class SetupCancelled(Exception):
    pass


class Prompter:
    """Numbered choices and questions over any pair of text streams."""

    def __init__(self, inp: TextIO, out: TextIO):
        self.inp, self.out = inp, out

    def _line(self, prompt: str) -> str:
        self.out.write(prompt)
        self.out.flush()
        line = self.inp.readline()
        if not line:
            raise SetupCancelled
        return line.strip()

    def ask(self, question: str, default: str = "") -> str:
        hint = f" [{default}]" if default else ""
        while True:
            answer = self._line(f"{question}{hint}: ") or default
            if answer:
                return answer

    def secret(self, question: str) -> str:
        if self.inp is sys.stdin and sys.stdin.isatty():
            while True:
                try:
                    answer = getpass.getpass(f"{question}: ", stream=self.out).strip()
                except EOFError:
                    raise SetupCancelled from None
                if answer:
                    return answer
        return self.ask(question)

    def choose(self, question: str, options: list[str], default: int = 0) -> int:
        self.out.write(question + "\n")
        for i, option in enumerate(options, 1):
            self.out.write(f"  {i}. {option}\n")
        while True:
            answer = self._line(f"> [{default + 1}] ") or str(default + 1)
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            self.out.write(f"  choose 1-{len(options)}\n")

    def confirm(self, question: str, default: bool = True) -> bool:
        hint = "Y/n" if default else "y/N"
        while True:
            answer = self._line(f"{question} [{hint}] ").lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False


def llm_table(values: dict, hide_key: bool = False) -> str:
    """values as a TOML [llm] table, keys in a fixed order."""
    lines = ["[llm]"]
    for k in _KEY_ORDER:
        if k not in values:
            continue
        v = values[k]
        if k == "api_key" and hide_key:
            text = '"(hidden)"'
        elif isinstance(v, bool):
            text = "true" if v else "false"
        elif isinstance(v, str):
            text = json.dumps(v)          # a JSON string is a valid TOML basic string
        else:
            text = repr(v)
        lines.append(f"{k} = {text}")
    return "\n".join(lines) + "\n"


def replace_llm_table(text: str, table: str) -> str | None:
    """text with its [llm] table replaced by table, or table appended if
    there is none; None if llm keys are set some other way (dotted keys,
    an inline table), which is left for the user to edit."""
    m = _LLM_HEADER.search(text)
    if m is None:
        if "llm" in tomllib.loads(text):
            return None
        new = text.rstrip("\n") + ("\n\n" if text.strip() else "") + table
    else:
        nxt = _ANY_HEADER.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        rest = text[end:]
        new = text[:m.start()] + table + ("\n" + rest if rest else "")
    try:
        if tomllib.loads(new).get("llm") != tomllib.loads(table)["llm"]:
            return None
    except tomllib.TOMLDecodeError:
        return None
    return new


def save_config(path: Path, values: dict, p: Prompter, secret: bool) -> bool:
    """Write values as the [llm] table of path, asking before replacing one."""
    out, table = p.out, llm_table(values)
    shown = llm_table(values, hide_key=True)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        try:
            current = tomllib.loads(text).get("llm")
            new_text = replace_llm_table(text, table)
        except tomllib.TOMLDecodeError as e:
            out.write(f"{path} is not valid TOML ({e}); not editing it.\n")
            current, new_text = None, None
        if new_text is None:
            out.write(f"Add this to {path} by hand"
                      f"{' (with your key in place of (hidden))' if secret else ''}:\n\n{shown}\n")
            return False
        if current is not None:
            out.write(f"{path} already has:\n\n{llm_table(current, hide_key=True)}\n"
                      f"Replace it with:\n\n{shown}\n")
            if not p.confirm("Replace?", default=False):
                out.write("Left unchanged.\n")
                return False
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_text = table
    path.write_text(new_text, encoding="utf-8")
    if secret:
        os.chmod(path, 0o600)
    out.write(f"Wrote {path}\n")
    return True


def _ask_url(p: Prompter) -> str:
    while True:
        url = p.ask("Base URL (ending in /v1)").strip().rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        if "://" not in url:
            p.out.write("  include the scheme, e.g. http://gpu-box:8000/v1\n")
            continue
        if not url.endswith("/v1") and p.confirm(f"  Add /v1, giving {url}/v1?", default=True):
            url += "/v1"
        return url


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _setup(app: str, p: Prompter) -> int:
    out = p.out
    ollama = ollama_endpoint()
    ollama_models = (list_models(ollama) or []) if ollama else []
    ollama_label = ("Local Ollama (not running)" if ollama is None
                    else f"Local Ollama (found: {_plural(len(ollama_models), 'model')})")
    kind = p.choose("Where is your model?",
                    [ollama_label, "A hosted API (OpenAI-compatible)",
                     "A self-hosted server (vLLM, LiteLLM, llama.cpp)"],
                    default=0 if ollama_models else 1)
    values: dict = {}
    secret = False
    if kind == 0:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if ollama is None:
            out.write(f"Ollama is not answering at {host}. Start it (`ollama serve`), pull a "
                      "model (`ollama pull <name>`), and run setup again.\n")
            return 1
        if not ollama_models:
            out.write("Ollama is running but has no models. Pull one (`ollama pull <name>`) "
                      "and run setup again.\n")
            return 1
        ep = Endpoint(ollama.url, ollama_models[p.choose("Model:", ollama_models)], name="ollama")
    else:
        url = _ask_url(p)
        ep = Endpoint(url, "")
        k = p.choose("API key:", ["From an environment variable (recommended)",
                                  "Paste it (saved in the config file, mode 0600)", "None"],
                     default=0 if kind == 1 else 2)
        if k == 0:
            while not _VAR_NAME.match(var := p.ask("Variable name")):
                out.write("  a variable name is letters, digits and _\n")
            values["api_key_env"] = var
            ep.key = os.environ.get(var, "")
            out.write(f"  {var} is set ✓\n" if ep.key else
                      f"  {var} is not set in this shell; testing without a key. "
                      f"Set it before using {app}.\n")
        elif k == 1:
            ep.key = values["api_key"] = p.secret("API key")
            secret = True
        models = list_models(ep)
        if models is None:
            out.write(f"  ✗ could not list models at {url}/models\n")
            if not p.confirm("Continue anyway?", default=False):
                return 1
            models = []
        else:
            out.write(f"  ✓ reachable; lists {_plural(len(models), 'model')}\n")
        options = models + ["Other (type a name)"]
        choice = p.choose("Model:", options) if models else len(models)
        ep.model = options[choice] if choice < len(models) else p.ask("Model name")
    values["url"], values["model"] = ep.url, ep.model

    out.write("Testing a request... ")
    out.flush()
    ep.thinking_switch = kind == 2
    t0 = time.monotonic()
    try:
        ep.chat(PING, "ping", timeout=min(ep.timeout, 30))
    except LLMError as e:
        out.write(f"✗\n  {e}\n")
        if not p.confirm("Save anyway?", default=False):
            return 1
    else:
        out.write(f"✓ {time.monotonic() - t0:.1f} s\n")
        if kind == 2:
            accepted = "chat_template_kwargs" not in ep._dropped
            values["thinking_switch"] = accepted
            out.write("  server accepts enable_thinking → thinking_switch = true\n" if accepted
                      else "  server rejects enable_thinking → thinking_switch = false\n")

    shared, own = config_home() / "adjudicate" / "config.toml", config_home() / app / "config.toml"
    where = p.choose("Save to:", [f"{shared} (shared by every tool that uses adjudicate)",
                                  f"{own} ({app} only)"])
    if not save_config((shared, own)[where], values, p, secret):
        return 1
    out.write("\n")
    return check_endpoint(app, out)


def run_setup(app: str, inp: TextIO | None = None, out: TextIO | None = None,
              require_tty: bool = True) -> int:
    """Ask where the model is, test it, and save the [llm] table."""
    inp, out = inp or sys.stdin, out or sys.stdout
    if require_tty and not inp.isatty():
        out.write(f"{app} --setup asks questions, so it needs a terminal. To configure "
                  f"without one, write the [llm] table by hand (see `{app} --help`).\n")
        return 2
    try:
        return _setup(app, Prompter(inp, out))
    except (SetupCancelled, KeyboardInterrupt):
        out.write("\nSetup cancelled; nothing was written.\n")
        return 1


def check_endpoint(app: str, out: TextIO | None = None, preset: str = "auto",
                   url: str | None = None, model: str | None = None,
                   timeout: float | None = None) -> int:
    """Print each resolved setting and where it came from, then test the endpoint."""
    out = out or sys.stdout
    try:
        ep = resolve_endpoint(app, preset, url=url, model=model, timeout=timeout)
    except ConfigError as e:
        out.write(f"{app}: {e}\n")
        return 2
    if ep is None:
        why = ("--llm none: no model is used." if preset == "none" else
               "No endpoint: nothing is configured and no local Ollama with a model is running.")
        out.write(f"{why}\nRun `{app} --setup` to configure one.\n")
        return 1
    shown = {"url": ep.url, "model": ep.model,
             "api_key": "(hidden)" if ep.key else "(none)",
             "timeout": f"{ep.timeout:g}", "thinking_switch": str(ep.thinking_switch).lower()}
    width = max(len(v) for v in shown.values())
    for name in ROWS:
        out.write(f"{name:<16} {shown[name]:<{width}}  {ep.sources.get(name, '')}".rstrip() + "\n")
    models = list_models(ep)
    listing = "✗" if models is None else f"✓ {_plural(len(models), 'model')}"
    t0 = time.monotonic()
    try:
        ep.chat(PING, "ping", timeout=min(ep.timeout, 30))
    except LLMError as e:
        out.write(f"/models {listing}   chat ✗ {e}\n")
        return 1
    out.write(f"/models {listing}   chat ✓ {time.monotonic() - t0:.1f} s\n")
    return 0
