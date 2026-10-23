from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Isotonic:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Isotonic:
        self.iso.fit(np.asarray(raw, float), np.asarray(y, float))
        return self

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self.iso.predict(np.asarray(raw, float))
