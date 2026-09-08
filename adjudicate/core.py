"""Ask a model to choose between supplied candidates for marked spans.

The contract: each item carries a short list of options (the last usually
KEEP), a rule of thumb, and the sentence with the span marked [[...]].
The model picks one option per item. Anything else is rejected."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .llm import Endpoint, LLMError

SYSTEM_TEMPLATE = """{task}
For each item, choose exactly one option from its numbered list. {keep_rule}
Answer with the option's number (or the word KEEP); never retype or paraphrase the option text.
Respond with JSON only, in this shape:
{{"decisions": [{{"id": 1, "choice": "<option number, or KEEP>", "confidence": 0.0, "reason": "<under 12 words>"}}]}}"""

DEFAULT_KEEP_RULE = ("Choose KEEP when the marked text is already correct in context, is part of a "
                     "proper name, a title, a quotation, a URL or code, or when no option fits.")


def system_prompt(task: str, keep_rule: str = DEFAULT_KEEP_RULE) -> str:
    return SYSTEM_TEMPLATE.format(task=task, keep_rule=keep_rule)


@dataclass
class Item:
    id: int
    word: str
    options: tuple[str, ...]
    rule: str
    context: str
    line: int
    start: int
    end: int
    source: str = "ambiguous"     # caller-defined tag for reporting


@dataclass
class Decision:
    item: Item
    choice: str | None            # None = could not decide
    confidence: float
    reason: str
    cached: bool = False

    @property
    def replacement(self) -> str | None:
        """The text to substitute, or None to leave the word alone."""
        if self.choice is None or self.choice == "KEEP":
            return None
        if self.choice.lower() == self.item.word.lower():
            return None
        return self.choice


class Cache:
    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, dict] = {}
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    @staticmethod
    def key(scope: str, item: Item) -> str:
        """scope: anything that changes the right answer (target variety, task)."""
        norm = re.sub(r"\s+", " ", item.context).strip().lower()
        h = hashlib.sha256(f"{scope}|{item.word.lower()}|{'|'.join(item.options)}|{norm}".encode()).hexdigest()
        return h[:24]

    def get(self, k: str) -> dict | None:
        return self.data.get(k)

    def put(self, k: str, d: Decision) -> None:
        self.data[k] = {"word": d.item.word, "choice": d.choice,
                        "confidence": d.confidence, "reason": d.reason,
                        "context": d.item.context}

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))


def render_batch(items: list[Item]) -> str:
    parts = []
    for it in items:
        opts = " | ".join(o if o == "KEEP" else f"[{n}] {o}" for n, o in enumerate(it.options, 1))
        parts.append(
            f"Item {it.id} — word: \"{it.word}\" — options: {opts}\n"
            f"Rule: {it.rule}\n"
            f"Context: {it.context}")
    return "\n\n".join(parts)


def match_choice(choice: str, options: tuple[str, ...]) -> str | None:
    """Resolve the model's answer to one option: by number ("2", "[2]"),
    by the word KEEP, or by the option text (trimmed, case-insensitive)."""
    c = str(choice).strip()
    m = re.fullmatch(r"\[?(\d+)\]?", c)
    if m:
        n = int(m.group(1))
        return options[n - 1] if 1 <= n <= len(options) else None
    if c.upper() == "KEEP" and "KEEP" in options:
        return "KEEP"
    c = re.sub(r"^\[\d+\]\s*", "", c)
    return next((o for o in options if o.strip().lower() == c.lower()), None)


def parse_response(text: str) -> dict[int, dict]:
    """Return {id: {choice, confidence, reason}} from the model's reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    m = re.search(r"[\[{]", text)
    if not m:
        return {}
    try:
        data = json.loads(text[m.start():])
    except json.JSONDecodeError:
        # Try to salvage the largest balanced object.
        end = max(text.rfind("}"), text.rfind("]"))
        try:
            data = json.loads(text[m.start():end + 1])
        except json.JSONDecodeError:
            return {}
    if isinstance(data, dict):
        data = data.get("decisions", data.get("items", []))
    out = {}
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and "id" in d:
            try:
                out[int(d["id"])] = d
            except (TypeError, ValueError):
                continue
    return out


def decide_batch(ep: Endpoint, system: str, items: list[Item]) -> list[Decision]:
    try:
        reply = ep.chat(system, render_batch(items))
    except LLMError as e:
        return [Decision(it, None, 0.0, f"model error: {e}") for it in items]
    answers = parse_response(reply)
    out = []
    for it in items:
        a = answers.get(it.id)
        if not a:
            out.append(Decision(it, None, 0.0, "no answer from model"))
            continue
        choice = str(a.get("choice", "")).strip()
        matched = match_choice(choice, it.options)
        try:
            conf = float(a.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        reason = str(a.get("reason", ""))[:200]
        if matched is None:
            out.append(Decision(it, None, 0.0, f"invalid choice {choice!r}"))
        else:
            out.append(Decision(it, matched, conf, reason))
    return out


def adjudicate(ep: Endpoint | None, scope: str, items: list[Item], cache: Cache,
               batch_size: int = 20, parallel: int = 4, verbose: bool = False,
               system: str | None = None, log_name: str = "adjudicate") -> list[Decision]:
    """Decide every item: from cache where possible, otherwise from the model
    in parallel batches. `scope` keys the cache; `system` is the task prompt
    (see system_prompt)."""
    system = system or system_prompt("You are a meticulous copy editor.")
    decisions: dict[int, Decision] = {}
    todo: list[Item] = []
    for it in items:
        k = Cache.key(scope, it)
        hit = cache.get(k)
        if hit and hit.get("choice") in it.options:
            decisions[it.id] = Decision(it, hit["choice"], float(hit.get("confidence", 1.0)),
                                        hit.get("reason", ""), cached=True)
        else:
            todo.append(it)
    if todo and ep is None:
        for it in todo:
            decisions[it.id] = Decision(it, None, 0.0, "no model configured")
    elif todo:
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        if verbose:
            print(f"{log_name}: asking {ep.name} ({ep.model}) about {len(todo)} items "
                  f"in {len(batches)} batch(es), {parallel} in parallel", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            for result in pool.map(lambda b: decide_batch(ep, system, b), batches):
                for d in result:
                    decisions[d.item.id] = d
                    if d.choice is not None:
                        cache.put(Cache.key(scope, d.item), d)
        cache.save()
    return [decisions[it.id] for it in items]
