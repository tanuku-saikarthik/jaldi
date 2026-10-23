from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.nan_to_num(s, nan=-1e9)))


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def brier(y, p) -> float:
    return float(brier_score_loss(np.asarray(y), np.clip(np.asarray(p, float), 0, 1)))


def reliability(y, p, bins: int = 10) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = {"bin_mean_p": [], "bin_frac_pos": [], "bin_n": []}
    for b in range(bins):
        m = idx == b
        out["bin_n"].append(int(m.sum()))
        out["bin_mean_p"].append(float(p[m].mean()) if m.any() else None)
        out["bin_frac_pos"].append(float(y[m].mean()) if m.any() else None)
    return out


def grouped_bootstrap(groups, stat_fn, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% CI, resampling whole groups (turns) with replacement.

    groups: list of per-group payloads; stat_fn(list_of_payloads) -> float
    """
    rng = np.random.default_rng(seed)
    point = stat_fn(groups)
    k = len(groups)
    vals = []
    for _ in range(n):
        sample = [groups[i] for i in rng.integers(0, k, k)]
        v = stat_fn(sample)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mean_ci(values, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    vals = [float(v) for v in values]
    return grouped_bootstrap(vals, lambda xs: float(np.mean(xs)) if xs else float("nan"), n, seed)
