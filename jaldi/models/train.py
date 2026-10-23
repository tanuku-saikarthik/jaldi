"""Stage 6: train, calibrate and evaluate P(safe) models for L1 and L2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..eval.metrics import auroc, brier, ece, grouped_bootstrap, reliability
from ..features.build import ALL_FEATURES, GROUPS
from ..io import write_json
from .calibrate import Isotonic
from .fusion import attach_encoder_scores
from .gbm import predict, train_gbm

N_SEEDS = 5
# single-signal heuristics used as scorers for the AUROC gate (higher = more likely safe)
HEURISTIC_SCORES = {
    "pause (silence_ms)": "silence_ms",
    "stable (ms_since_change)": "ms_since_change",
    "stable (stable_count)": "stable_count",
    "turn detector (turn_p)": "turn_p",
}


def usable_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Drop features that are all-NaN in this run (e.g. prosody in text-sim mode)."""
    return [f for f in features if f in df and df[f].notna().any()]


def _auroc_ci(test: pd.DataFrame, score_col: str, label: str, cfg: Config):
    groups = [(g[label].values, g[score_col].values) for _, g in test.groupby("id")]

    def stat(gs):
        y = np.concatenate([a for a, _ in gs])
        s = np.concatenate([b for _, b in gs])
        return auroc(y, s)

    return grouped_bootstrap(groups, stat, cfg.bootstrap, cfg.seed)


def train_all(df: pd.DataFrame, cfg: Config, enc_path: str) -> tuple[pd.DataFrame, dict]:
    df, enc_cols = attach_encoder_scores(df, enc_path)
    feats = usable_features(df, ALL_FEATURES)
    dropped = [f for f in ALL_FEATURES if f not in feats]
    tr, va, te = (df[df.split == s].copy() for s in ("train", "val", "test"))
    out = {"features_used": feats, "features_dropped_all_nan": dropped, "encoder_fusion": bool(enc_cols),
           "n_rows": {"train": len(tr), "val": len(va), "test": len(te)},
           "n_turns": {s: int(df[df.split == s].id.nunique()) for s in ("train", "val", "test")},
           "labels": {}}
    preds = df[["id", "idx", "split", "lang", "t_ms", "l1", "l2"]].copy()

    for label in ("l1", "l2"):
        fs = feats + ([f"enc_{label}"] if f"enc_{label}" in enc_cols else [])
        seed_raw_val, seed_raw_all, seed_auc = [], [], []
        importance = np.zeros(len(fs))
        model_dir = cfg.data / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for s in range(N_SEEDS):
            m = train_gbm(tr, va, fs, label, seed=cfg.seed + s)
            m.save_model(str(model_dir / f"{label}_seed{s}.txt"), num_iteration=m.best_iteration)
            rv, ra = predict(m, va, fs), predict(m, df, fs)
            seed_raw_val.append(rv)
            seed_raw_all.append(ra)
            seed_auc.append(auroc(te[label], predict(m, te, fs)))
            importance += m.feature_importance("gain")
        raw_val = np.mean(seed_raw_val, 0)
        raw_all = np.mean(seed_raw_all, 0)
        cal = Isotonic().fit(raw_val, va[label].values)
        write_json(model_dir / f"{label}_meta.json", {
            "features": fs, "n_seeds": N_SEEDS,
            "isotonic_x": cal.iso.X_thresholds_.tolist(), "isotonic_y": cal.iso.y_thresholds_.tolist(),
        })
        preds[f"raw_{label}"] = raw_all
        preds[f"p_{label}"] = cal(raw_all)
        te_p = preds.loc[te.index, f"p_{label}"].values
        te = te.assign(**{f"p_{label}": te_p})

        res = {
            "base_rate_test": float(te[label].mean()),
            "auroc_seeds": [float(a) for a in seed_auc],
            "auroc_seed_mean": float(np.mean(seed_auc)),
            "auroc_seed_std": float(np.std(seed_auc)),
            "auroc_ci": _auroc_ci(te, f"p_{label}", label, cfg),
            "ece_test": ece(te[label], te_p),
            "brier_test": brier(te[label], te_p),
            "ece_test_uncalibrated": ece(te[label], preds.loc[te.index, f"raw_{label}"].values),
            "reliability_test": reliability(te[label], te_p),
            "by_lang": {},
            "heuristics": {},
            "top_features": sorted(zip(fs, importance.tolist()), key=lambda x: -x[1])[:12],
        }
        for lang, g in te.groupby("lang"):
            res["by_lang"][lang] = {
                "auroc": auroc(g[label], g[f"p_{label}"]),
                "ece": ece(g[label], g[f"p_{label}"]),
                "brier": brier(g[label], g[f"p_{label}"]),
                "base_rate": float(g[label].mean()),
                "n_rows": len(g),
            }
        for name, col in HEURISTIC_SCORES.items():
            if col in te and te[col].notna().any():
                res["heuristics"][name] = {"auroc_ci": _auroc_ci(te, col, label, cfg)}
        best_name, best = None, None
        for name, h in res["heuristics"].items():
            if best is None or h["auroc_ci"][0] > best["auroc_ci"][0]:
                best_name, best = name, h
        res["gate"] = {
            "best_heuristic": best_name,
            "best_heuristic_auroc_ci": best["auroc_ci"] if best else None,
            "model_auroc_ci": res["auroc_ci"],
            "passed": bool(best is None or res["auroc_ci"][1] > best["auroc_ci"][2]),
            "rule": "model 95% CI lower bound > best single heuristic 95% CI upper bound",
        }
        # ablation: drop one feature group at a time (single seed, test AUROC)
        abl = {}
        for g, gfs in GROUPS.items():
            keep = [f for f in fs if f not in gfs]
            if len(keep) == len(fs):
                continue
            m = train_gbm(tr, va, keep, label, seed=cfg.seed)
            abl[f"without_{g}"] = auroc(te[label], predict(m, te, keep))
        m = train_gbm(tr, va, fs, label, seed=cfg.seed)
        abl["all_features"] = auroc(te[label], predict(m, te, fs))
        res["ablation_test_auroc"] = abl
        out["labels"][label] = res
    return preds, out
