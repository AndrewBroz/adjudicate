"""Sentence context, position heuristics, and Markdown prose regions."""

from __future__ import annotations

import re

from dataclasses import dataclass  # noqa: F401

FENCE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.M | re.S)
_INLINE_CODE = re.compile(r"(`+)[^`\n]*?\1")
_LINK_DEST = re.compile(r"\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
_URL = re.compile(r"<?(?:https?|ftp|mailto):[^\s>)]+>?")
FRONT_MATTER = re.compile(r"\A\ufeff?---\n.*?\n---\n", re.S)  # an optional byte-order mark first
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>\n]*>")
HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")

# Obsidian syntax. Tags start with a letter so "#42" and headings ("# x") stay prose;
# inline maths needs non-space just inside both dollars and no digit after the
# closing one, so "$5 and $10" stays prose.
WIKILINK = re.compile(r"!?\[\[[^\[\]\n]+\]\]")
OBSIDIAN_COMMENT = re.compile(r"%%.*?%%", re.S)
TAG = re.compile(r"(?<![\w&/#])#[A-Za-z][\w/-]*")
BLOCK_ID = re.compile(r"(?<=\s)\^[A-Za-z0-9-]+[ \t]*$", re.M)
CALLOUT = re.compile(r"^[ \t]*(?:>[ \t]*)+\[![^\]\n]+\][+-]?", re.M)
MATH_BLOCK = re.compile(r"\$\$.*?\$\$", re.S)
MATH_INLINE = re.compile(r"(?<![\\$\w])\$(?=\S)[^$\n]*?(?<=\S)\$(?![\w$])")

# Pre-0.3 names, kept so an old consumer does not break mid-upgrade; removed in 0.4.
_FENCE, _FRONT, _HTML_ENTITY = FENCE, FRONT_MATTER, HTML_ENTITY


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


_LIST_ITEM = re.compile(r"[ \t]*(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)")


def indented_code(text: str) -> list[tuple[int, int]]:
    """Indented code blocks: lines indented four spaces or a tab, after a blank line (or
    at the start), not inside a list (where such a line continues the item)."""
    out: list[tuple[int, int]] = []
    pos, start, prev_blank, in_list = 0, None, True, False
    for ln in text.splitlines(keepends=True):
        body = ln.rstrip("\r\n")
        blank = not body.strip()
        indented = not blank and body.startswith(("    ", "\t"))
        if start is not None:
            if indented:
                pos += len(ln)
                continue
            out.append((start, pos))
            start = None
        if indented and prev_blank and not in_list:
            start = pos
        elif not blank:
            if _LIST_ITEM.match(body):
                in_list = True
            elif not body[:1].isspace() and prev_blank:
                in_list = False             # a new paragraph after a blank line ends the list
        prev_blank = blank
        pos += len(ln)
    if start is not None:
        out.append((start, pos))
    return out


def protected_spans(text: str, entities: bool = True) -> list[tuple[int, int]]:
    """Spans no tool should edit: front matter, fenced, indented and inline code, link
    targets, URLs, HTML comments and tags, HTML entities, and Obsidian syntax
    (wikilinks/embeds, comments, tags, block ids, callout markers, inline and
    block maths)."""
    spans = []
    m = FRONT_MATTER.match(text)
    if m:
        spans.append((0, m.end()))
    rxs = [FENCE, _INLINE_CODE, _LINK_DEST, _URL, _HTML_COMMENT, _HTML_TAG,
           WIKILINK, OBSIDIAN_COMMENT, TAG, BLOCK_ID, CALLOUT, MATH_BLOCK, MATH_INLINE]
    if entities:
        rxs.append(HTML_ENTITY)
    for rx in rxs:
        spans.extend(m.span() for m in rx.finditer(text))
    spans.extend(indented_code(text))
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


_APOSTROPHE = re.compile(r"(?<=\w)’(?=\w)")


def inside_quotes(text: str, pos: int) -> bool:
    """Cheap check: an odd number of opening quotes precede `pos` in the paragraph."""
    ps, _ = paragraph_bounds(text, pos)
    seg = text[ps:pos]
    opens = seg.count("“") + seg.count("‘") + len(re.findall(r'(?:^|\s)["\']', seg))
    # `seg` ends where the word starts, so a quote at its very end is an opener.
    # A curled apostrophe between letters (Ministry’s, don’t) closes nothing.
    closes = (seg.count("”") + seg.count("’") - len(_APOSTROPHE.findall(seg))
              + len(re.findall(r'["\'](?=[\s.,;:!?)])', seg)))
    return opens > closes
