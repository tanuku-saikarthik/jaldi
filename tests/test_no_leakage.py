"""Hard rule 2: no feature may read the future.

For every partial i, features computed with the full turn must equal features computed when the
turn is truncated right after partial i, and changing anything after i must not change them.
"""

import copy
import math
import random

from jaldi.asr.text_sim import simulate_turn
from jaldi.config import Config
from jaldi.corpus.generate import generate
from jaldi.features.build import ALL_FEATURES, partial_features


def _eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _turns(n=40):
    cfg = Config(n_turns=n)
    return [(t, simulate_turn(t, cfg)) for t in generate(cfg)]


def test_features_equal_on_truncated_turns():
    for t, rec in _turns():
        ps = rec["partials"]
        for i in range(len(ps)):
            full = partial_features(ps, i, t["lang"])
            trunc = partial_features(ps[: i + 1], i, t["lang"])
            assert set(full) == set(ALL_FEATURES)
            for k in ALL_FEATURES:
                assert _eq(full[k], trunc[k]), (t["id"], i, k)


def test_future_edits_do_not_change_features():
    rng = random.Random(0)
    for t, rec in _turns(20):
        ps = rec["partials"]
        for i in range(len(ps) - 1):
            before = partial_features(ps, i, t["lang"])
            altered = copy.deepcopy(ps)
            for p in altered[i + 1:]:
                p["text"] = p["text"] + " नहीं"
                p["silence_ms"] = rng.uniform(0, 2000)
                p["t_ms"] += 1000
            after = partial_features(altered, i, t["lang"])
            for k in ALL_FEATURES:
                assert _eq(before[k], after[k]), (t["id"], i, k)
