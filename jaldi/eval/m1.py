"""Milestone M1: how often does meaning flip late, per language, and what does a pause heuristic waste?"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import Config
from ..sim import policy as P
from ..sim.simulate import (
    evaluate,
    has_meaning_flip,
    is_false_actionable,
    make_view,
    oracle_headroom,
)
from .metrics import mean_ci

LANGS = ("en", "hi", "hinglish")


def _words_after_settle(turn: dict) -> int:
    """Words in the final transcript after the action last became (and stayed) final."""
    ps = turn["partials"]
    j = len(ps)
    while j > 0 and ps[j - 1]["l2"]:
        j -= 1
    if j == len(ps):
        return 0
    n_final = len(turn["final_text"].split())
    return max(0, n_final - len(ps[j]["text"].split()))


def m1_analysis(turns: list[dict], cfg: Config) -> dict:
    out: dict = {"by_lang": {}, "flip_by_phenomenon": {}, "pause_sweep_test_l2": {},
                 "note": "Descriptive stats use all turns; the heuristic sweep uses the test split."}
    for lang in LANGS:
        ts = [t for t in turns if t["lang"] == lang]
        flip = mean_ci([has_meaning_flip(t) for t in ts], cfg.bootstrap, cfg.seed)
        fa = mean_ci([is_false_actionable(t) for t in ts], cfg.bootstrap, cfg.seed)
        out["by_lang"][lang] = {
            "n_turns": len(ts),
            "meaning_flip_rate": flip,
            "false_actionable_rate": fa,
            "l1_headroom_ms_median": float(np.median([oracle_headroom(t, "l1") for t in ts])),
            "l2_headroom_ms_median": float(np.median([oracle_headroom(t, "l2") for t in ts])),
            "l2_headroom_ms_mean": mean_ci([oracle_headroom(t, "l2") for t in ts], cfg.bootstrap, cfg.seed),
            "words_after_action_settles_mean": float(np.mean([_words_after_settle(t) for t in ts])),
        }
        by = defaultdict(list)
        for t in ts:
            key = t["phenomenon"]
            if key == "late_correction":
                key = f"late_correction_{t['correction_placement']}"
            by[key].append(has_meaning_flip(t))
        out["flip_by_phenomenon"][lang] = {k: {"rate": float(np.mean(v)), "n": len(v)}
                                           for k, v in sorted(by.items())}

    test = [t for t in turns if t["split"] == "test"]
    for lang in LANGS:
        views = [make_view(t) for t in test if t["lang"] == lang]
        sweep = []
        for x in cfg.pause_sweep_ms:
            r = evaluate(views, [P.pause(v, x) for v in views], "l2", cfg.time_to_ready_ms,
                         cfg.max_speculations, cfg.endpoint_delay_ms)
            sweep.append({"x_ms": x, **r})
        out["pause_sweep_test_l2"][lang] = sweep
    return out
