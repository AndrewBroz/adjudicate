"""Changes between texts as labelled, non-overlapping splices.

A pass's changes are derived from a diff of its input and output (lines, then
tokens inside changed lines), so applying them always reproduces the output;
the pass's own records only label the hunks. compose() re-expresses a second
pass's changes against the first pass's input, merging any that overlap a
first-pass change into one change that depends on (and supersedes) it."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"\w+|\s+|[^\w\s]")


@dataclass
class Change:
    start: int
    end: int
    before: str
    after: str
    check: str = ""
    reason: str = ""
    decided_by: str = "rule"          # rule | model | cache
    confidence: float = 1.0
    id: str = ""
    pass_: str = ""
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Annotation:
    before: str
    after: str
    check: str
    reason: str = ""
    decided_by: str = "rule"
    confidence: float = 1.0


@dataclass(frozen=True)
class ReviewItem:
    line: int
    kind: str                          # undecided | rejected | alert | note
    message: str
    check: str = ""
    pass_: str = ""


@dataclass(frozen=True)
class Skipped:
    what: str
    why: str
    fix: str


def _starts(parts: list[str]) -> list[int]:
    out = [0]
    for p in parts:
        out.append(out[-1] + len(p))
    return out


def _token_hunks(a: str, b: str, base: int):
    ta, tb = _TOKEN.findall(a), _TOKEN.findall(b)
    oa, ob = _starts(ta), _starts(tb)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:      # one hunk per differing token pair
            for k in range(i2 - i1):
                if ta[i1 + k] != tb[j1 + k]:
                    yield base + oa[i1 + k], base + oa[i1 + k + 1], tb[j1 + k]
        else:
            yield base + oa[i1], base + oa[i2], b[ob[j1]:ob[j2]]


def _hunks(before: str, after: str):
    la, lb = before.splitlines(keepends=True), after.splitlines(keepends=True)
    pa, pb = _starts(la), _starts(lb)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:      # line pairs: avoids one huge token diff
            for k in range(i2 - i1):
                yield from _token_hunks(before[pa[i1 + k]:pa[i1 + k + 1]],
                                        after[pb[j1 + k]:pb[j1 + k + 1]], pa[i1 + k])
        else:
            yield from _token_hunks(before[pa[i1]:pa[i2]], after[pb[j1]:pb[j2]], pa[i1])


def derive_changes(before: str, after: str, annotations=(), default_check: str = "") -> list[Change]:
    """The changes that turn before into after, labelled from annotations
    (each used at most once, matched on exact before/after text, in order;
    failing that, on one of the annotation's own sub-hunks).
    Adjacent unlabelled hunks with no whitespace between them are joined, so
    one word's punctuation edit (both quotes of "hi") is one change."""
    pool = list(annotations)
    hunks = [Change(s, e, before[s:e], new, check=default_check) for s, e, new in _hunks(before, after)]
    hit = [False] * len(hunks)

    def label(c: Change, a) -> None:
        c.check, c.reason, c.decided_by, c.confidence = a.check, a.reason, a.decided_by, a.confidence

    for i, c in enumerate(hunks):                  # exact (before, after) first
        for k, a in enumerate(pool):
            if a.before == c.before and a.after == c.after:
                label(c, a)
                del pool[k]
                hit[i] = True
                break
    # Then an annotation's own minimal sub-hunks ('third party' -> 'third-party'
    # is the single hunk ' ' -> '-'); each sub-pair labels one hunk.
    subs = [(a, [(a.before[s:e], new) for s, e, new in _hunks(a.before, a.after)]) for a in pool]
    for i, c in enumerate(hunks):
        if hit[i]:
            continue
        for a, pairs in subs:
            if (c.before, c.after) in pairs:
                pairs.remove((c.before, c.after))
                label(c, a)
                hit[i] = True
                break
    out: list[Change] = []
    labelled: list[bool] = []
    for c, h in zip(hunks, hit):
        if (not h and out and not labelled[-1]
                and not any(ch.isspace() for ch in before[out[-1].end:c.start])):
            p = out[-1]
            p.after += before[p.end:c.start] + c.after
            p.end, p.before = c.end, before[p.start:c.end]
            continue
        out.append(c)
        labelled.append(h)
    return out


def apply_changes(text: str, changes: list[Change]) -> str:
    """Apply a selection: drop changes a selected change supersedes, then
    splice the rest from the end of the text backwards."""
    superseded = {d for c in changes for d in c.depends_on}
    keep = [c for c in changes if c.id not in superseded or not c.id]
    for c in sorted(keep, key=lambda c: (c.start, c.end), reverse=True):
        if text[c.start:c.end] != c.before:
            raise ValueError(f"change {c.id or '?'} at {c.start}: expected {c.before!r}, "
                             f"found {text[c.start:c.end]!r}")
        text = text[:c.start] + c.after + text[c.end:]
    return text


def _overlaps(a: int, b: int, ms: int, me: int) -> bool:
    if ms == me:                      # a deletion in the middle text: touching counts
        return a <= ms <= b
    if a == b:                        # an insertion: strictly inside counts
        return ms < a < me
    return a < me and ms < b


def compose(original: str, middle: str, first: list[Change], second: list[Change]) -> list[Change]:
    """first: changes original -> middle; second: changes middle -> final.
    Returns first plus second re-expressed against original."""
    F = sorted(first, key=lambda c: (c.start, c.end))
    mid, shift = [], 0
    for f in F:
        ms = f.start + shift
        mid.append((ms, ms + len(f.after), f))
        shift += len(f.after) - (f.end - f.start)

    def to_orig(p: int, exclude: set[int] = frozenset()) -> int:
        return p - sum(len(f.after) - (f.end - f.start)
                       for k, (ms, me, f) in enumerate(mid) if me <= p and k not in exclude)

    out = list(first)
    group_s: list[Change] = []
    group_f: set[int] = set()

    def flush():
        if not group_s:
            return
        if not group_f:
            for s in group_s:
                out.append(Change(to_orig(s.start), to_orig(s.end), s.before, s.after, s.check,
                                  s.reason, s.decided_by, s.confidence, s.id, s.pass_))
        else:
            ua = min([s.start for s in group_s] + [mid[k][0] for k in group_f])
            ub = max([s.end for s in group_s] + [mid[k][1] for k in group_f])
            seg = middle[ua:ub]
            for s in sorted(group_s, key=lambda s: (s.start, s.end), reverse=True):
                seg = seg[:s.start - ua] + s.after + seg[s.end - ua:]
            oa, ob = to_orig(ua, group_f), to_orig(ub)
            by = [s.decided_by for s in group_s]
            out.append(Change(oa, ob, original[oa:ob], seg,
                              "+".join(dict.fromkeys(s.check for s in group_s)),
                              "; ".join(r for r in dict.fromkeys(s.reason for s in group_s) if r),
                              "model" if "model" in by else "cache" if "cache" in by else "rule",
                              min(s.confidence for s in group_s), group_s[0].id, group_s[0].pass_,
                              tuple(mid[k][2].id for k in sorted(group_f))))
        group_s.clear()
        group_f.clear()

    # A group is a run of second-pass changes connected through the first-pass
    # changes they overlap; a second-pass change that overlaps none is its own group.
    for s in sorted(second, key=lambda c: (c.start, c.end)):
        hit = {k for k, (ms, me, f) in enumerate(mid)
               if _overlaps(s.start, s.end, ms, me)
               or (f.start == f.end and s.start == s.end and ms <= s.start <= me)}   # same-spot insertions
        if group_s and not (hit & group_f):
            flush()
        group_s.append(s)
        group_f |= hit
        if not hit:
            flush()
    flush()
    return out
