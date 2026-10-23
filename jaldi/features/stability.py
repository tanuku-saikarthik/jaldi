"""How much the streaming transcript is still moving, computed from partials up to i only."""

from __future__ import annotations

from ..text import tokens

STABILITY_FEATURES = [
    "edit_dist_last", "revision_last", "revisions_last5", "ms_since_change", "stable_count",
    "words_added_last", "asr_conf",
]


def word_edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def stability_features(partials: list[dict], i: int) -> dict:
    """Features for partial i using partials[0..i] only."""
    toks = [tokens(p["text"]) for p in partials[: i + 1]]
    cur = toks[i]
    prev = toks[i - 1] if i > 0 else []

    def is_revision(k: int) -> int:
        if k == 0:
            return 0
        a, b = toks[k - 1], toks[k]
        return int(a != b and b[: len(a)] != a)

    last_change = 0
    for k in range(i, 0, -1):
        if toks[k] != toks[k - 1]:
            last_change = k
            break
    stable = 0
    for k in range(i, 0, -1):
        if toks[k] == toks[k - 1]:
            stable += 1
        else:
            break
    return {
        "edit_dist_last": word_edit_distance(prev, cur),
        "revision_last": is_revision(i),
        "revisions_last5": sum(is_revision(k) for k in range(max(0, i - 4), i + 1)),
        "ms_since_change": partials[i]["t_ms"] - partials[last_change]["t_ms"],
        "stable_count": stable,
        "words_added_last": len(cur) - len(prev),
        "asr_conf": partials[i].get("asr_conf", float("nan")),
    }
