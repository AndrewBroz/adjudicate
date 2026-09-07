"""Splice replacements into text at Vale's (line, span) positions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edit:
    line: int      # 1-indexed
    start: int     # 1-indexed inclusive column
    end: int       # 1-indexed inclusive column
    original: str
    replacement: str
    source: str    # check name or "adjudicated"


def match_case(original: str, replacement: str) -> str:
    """Give `replacement` the casing pattern of `original`."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def line_offsets(text: str) -> list[int]:
    """Absolute offset of the start of each line (index 0 = line 1)."""
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def to_offsets(text: str, edit: Edit) -> tuple[int, int]:
    starts = line_offsets(text)
    base = starts[edit.line - 1]
    return base + edit.start - 1, base + edit.end


def apply_edits(text: str, edits: list[Edit]) -> tuple[str, list[Edit]]:
    """Apply edits whose span still contains the expected original text.

    Returns the new text and the edits that were skipped because the text
    at their position no longer matched.
    """
    skipped: list[Edit] = []
    spans = []
    for e in edits:
        s, t = to_offsets(text, e)
        if text[s:t] != e.original:
            skipped.append(e)
            continue
        spans.append((s, t, e))
    # Apply from the end so earlier offsets stay valid; drop overlaps.
    spans.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = text
    last_start = len(text) + 1
    for s, t, e in spans:
        if t > last_start:
            skipped.append(e)
            continue
        out = out[:s] + match_case(e.original, e.replacement) + out[t:]
        last_start = s
    return out, skipped
