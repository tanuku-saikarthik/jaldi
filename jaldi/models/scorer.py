"""Load a trained Jaldi model (seed ensemble + isotonic) and score partials online."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.build import partial_features
from ..io import read_json


class JaldiScorer:
    def __init__(self, model_dir: str | Path, label: str = "l2"):
        d = Path(model_dir)
        meta = read_json(d / f"{label}_meta.json")
        self.features = meta["features"]
        self.boosters = [lgb.Booster(model_file=str(d / f"{label}_seed{s}.txt")) for s in range(meta["n_seeds"])]
        self.iso_x, self.iso_y = np.array(meta["isotonic_x"]), np.array(meta["isotonic_y"])

    def score(self, partials: list[dict], lang: str) -> list[float]:
        """P(safe) for every partial; partial i only sees partials[0..i]."""
        rows = [partial_features(partials[: i + 1], i, lang) for i in range(len(partials))]
        X = pd.DataFrame(rows)[self.features]
        raw = np.mean([b.predict(X) for b in self.boosters], 0)
        return np.interp(raw, self.iso_x, self.iso_y).tolist()
