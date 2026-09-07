"""Sentence context, position heuristics, and Markdown prose regions."""

from __future__ import annotations

import re

from dataclasses import dataclass  # noqa: F401

_FENCE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.M | re.S)
_INLINE_CODE = re.compile(r"(`+)[^`\n]*?\1")
_LINK_DEST = re.compile(r"\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
_URL = re.compile(r"<?(?:https?|ftp|mailto):[^\s>)]+>?")
_FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>\n]*>")
_HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")



@dataclass
class _Unused:
    quotes: int = 0
    apostrophes: int = 0
    dashes: int = 0
    ranges: int = 0
    dates: int = 0
    punctuation_moved: int = 0

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if v}


def protected_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    m = _FRONT.match(text)
    if m:
        spans.append((0, m.end()))
    for rx in (_FENCE, _INLINE_CODE, _LINK_DEST, _URL, _HTML_COMMENT, _HTML_TAG, _HTML_ENTITY):
        spans.extend(m.span() for m in rx.finditer(text))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def prose_regions(text: str) -> list[tuple[int, int]]:
    regions, cursor = [], 0
    for s, e in protected_spans(text):
        if s > cursor:
            regions.append((cursor, s))
        cursor = e
    if cursor < len(text):
        regions.append((cursor, len(text)))
    return regions


def map_prose(text: str, fn) -> str:
    out, cursor = [], 0
    for s, e in prose_regions(text):
        out.append(text[cursor:s])
        out.append(fn(text[s:e]))
        cursor = e
    out.append(text[cursor:])
    return "".join(out)



_SENT_END = re.compile(r"(?<=[.!?])\s+")


def paragraph_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n\n", 0, pos)
    start = 0 if start < 0 else start + 2
    end = text.find("\n\n", pos)
    end = len(text) if end < 0 else end
    return start, end


def sentence_context(text: str, start: int, end: int, radius: int = 1,
                     max_chars: int = 400) -> str:
    """The sentence containing [start, end) plus `radius` neighbours, with
    the target span marked as [[...]]."""
    ps, pe = paragraph_bounds(text, start)
    para = text[ps:pe]
    rel_s, rel_e = start - ps, end - ps
    pieces, cursor = [], 0
    for m in _SENT_END.finditer(para):
        pieces.append((cursor, m.start()))
        cursor = m.end()
    pieces.append((cursor, len(para)))
    idx = next((i for i, (a, b) in enumerate(pieces) if a <= rel_s < b), len(pieces) - 1)
    lo = pieces[max(0, idx - radius)][0]
    hi = pieces[min(len(pieces) - 1, idx + radius)][1]
    snippet = para[lo:rel_s] + "[[" + para[rel_s:rel_e] + "]]" + para[rel_e:hi]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > max_chars:
        mark = snippet.find("[[")
        a = max(0, mark - max_chars // 2)
        snippet = ("…" if a else "") + snippet[a:a + max_chars] + "…"
    return snippet


# Brackets are not sentence starts: "[Center for X](url)" and "(Centre for Y)"
# keep their capital because of the name, so the word is adjudicated.
_SENTENCE_START_PUNCT = set('.!?:"“‘\'')


def at_sentence_start(text: str, pos: int) -> bool:
    """True when a capital at `pos` is explained by position, not by being a name."""
    line_start = text.rfind("\n", 0, pos) + 1
    before = text[line_start:pos].rstrip()
    if not before:
        return True
    # Markdown markers: headings, list bullets, numbered items, blockquotes.
    if re.fullmatch(r"(#{1,6}|[-*+]|\d+[.)]|>|\|)", before):
        return True
    return before[-1] in _SENTENCE_START_PUNCT


def inside_quotes(text: str, pos: int) -> bool:
    """Cheap check: an odd number of opening quotes precede `pos` in the paragraph."""
    ps, _ = paragraph_bounds(text, pos)
    seg = text[ps:pos]
    opens = seg.count("“") + seg.count("‘") + len(re.findall(r'(?:^|\s)["\']', seg))
    # `seg` ends where the word starts, so a quote at its very end is an opener.
    closes = seg.count("”") + seg.count("’") + len(re.findall(r'["\'](?=[\s.,;:!?)])', seg))
    return opens > closes
