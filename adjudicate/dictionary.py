"""Hunspell lookups (via spylls) over the shipped en_US and en_GB dictionaries.
Shared by stylefix and proofix.

The dictionaries are the SCOWL-derived ones distributed with LibreOffice
(see dict/LICENSE-*.txt). en_GB is Oxford-inclusive: it accepts both
-ise and -ize spellings. en_US is smaller and lists no British forms.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from spylls.hunspell import Dictionary

DICT_DIR = Path(__file__).parent / "dict"
VARIETY_FILE = {"us": "en_US", "uk": "en_GB"}


@lru_cache(maxsize=None)
def _dictionary(variety: str) -> Dictionary:
    return Dictionary.from_files(str(DICT_DIR / VARIETY_FILE[variety]))


@lru_cache(maxsize=200_000)
def known(word: str, variety: str) -> bool:
    """True if `word` (any case) is a valid word in the variety, including
    inflected and affixed forms."""
    if not word:
        return False
    return _dictionary(variety).lookup(word) or _dictionary(variety).lookup(word.lower())


def known_anywhere(word: str) -> bool:
    return known(word, "us") or known(word, "uk")
