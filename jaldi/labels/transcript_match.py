"""L1 label: the partial already equals the final transcript (speculation reusable verbatim)."""

from __future__ import annotations

from ..text import normalize


def l1_labels(partial_texts: list[str], final_text: str) -> list[int]:
    f = normalize(final_text)
    return [int(normalize(p) == f) for p in partial_texts]
