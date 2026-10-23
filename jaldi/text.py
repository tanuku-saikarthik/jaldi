"""Text normalization shared by labels, features and the rule extractor."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[\.,!?;:\"“”()\[\]{}…।॥\-–—]+")
_SPACE = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    """NFC, lowercase, drop punctuation (incl. danda), unify nukta variants, collapse spaces."""
    t = unicodedata.normalize("NFC", text or "")
    # treat precomposed nukta letters and base+nukta the same (फ़ == फ़ -> फ)
    t = unicodedata.normalize("NFD", t).replace("़", "")
    t = unicodedata.normalize("NFC", t)
    t = t.replace("’", "'").replace("‘", "'").lower()
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    n = normalize(text)
    return n.split() if n else []


def is_devanagari(tok: str) -> bool:
    return bool(_DEVANAGARI.search(tok))


def codemix_index(text: str) -> float:
    """Share of the minority script among script-bearing tokens (0 = monolingual script)."""
    toks = tokens(text)
    dev = sum(1 for t in toks if _DEVANAGARI.search(t))
    lat = sum(1 for t in toks if _LATIN.search(t))
    tot = dev + lat
    return 0.0 if tot == 0 else min(dev, lat) / tot
