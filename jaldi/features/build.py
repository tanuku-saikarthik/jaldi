"""Stage 5: one feature row per partial, using only information available at that partial."""

from __future__ import annotations

import pandas as pd

from .acoustic import ACOUSTIC_FEATURES, TURN_FEATURES, acoustic_features, turn_features
from .stability import STABILITY_FEATURES, stability_features
from .text import TEXT_FEATURES, text_features

META_FEATURES = ["lang_en", "lang_hi", "lang_hinglish"]

GROUPS = {
    "text": TEXT_FEATURES,
    "stability": STABILITY_FEATURES,
    "acoustic": ACOUSTIC_FEATURES,
    "turn_detector": TURN_FEATURES,
    "meta": META_FEATURES,
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]


def partial_features(partials: list[dict], i: int, lang: str) -> dict:
    """Features of partial i. Reads partials[0..i] only (enforced by tests)."""
    p = partials[i]
    tf = text_features(p["text"])
    return {
        **tf,
        **stability_features(partials, i),
        **acoustic_features(p, tf["n_words"]),
        **turn_features(p),
        "lang_en": int(lang == "en"),
        "lang_hi": int(lang == "hi"),
        "lang_hinglish": int(lang == "hinglish"),
    }


def build_features(turns: list[dict]) -> pd.DataFrame:
    rows = []
    for t in turns:
        ps = t["partials"]
        for i in range(len(ps)):
            rows.append({
                "id": t["id"], "lang": t["lang"], "split": t["split"], "idx": i,
                "t_ms": ps[i]["t_ms"], "l1": ps[i]["l1"], "l2": ps[i]["l2"],
                **partial_features(ps[:i + 1], i, t["lang"]),
            })
    return pd.DataFrame(rows)
