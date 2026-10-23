"""Sweep every policy family, tune on val under a waste budget, evaluate on test per language."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from . import policy as P
from .simulate import evaluate, make_view

TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
LANGS = ("en", "hi", "hinglish")


def _configs(cfg: Config, has_turn_p: bool) -> list[tuple[str, float | None]]:
    cs = [("never", None), ("every", None)]
    cs += [("pause", x) for x in cfg.pause_sweep_ms]
    cs += [("stable", k) for k in cfg.stable_sweep_k]
    if has_turn_p:
        cs += [("turn_detector", t) for t in TAUS]
    cs += [("jaldi", t) for t in TAUS]
    return cs


def _trigger(family: str, param, view, scores: np.ndarray) -> np.ndarray:
    if family == "never":
        return P.never(view)
    if family == "every":
        return P.every(view)
    if family == "jaldi":
        return P.jaldi(scores, param)
    return P.FAMILIES[family](view, param)


def run_sweep(turns: list[dict], preds: pd.DataFrame, cfg: Config) -> dict:
    views = {t["id"]: make_view(t) for t in turns}
    meta = {t["id"]: (t["split"], t["lang"]) for t in turns}
    has_turn_p = any(np.isfinite(v.turn_p).any() for v in views.values())
    pr = preds.sort_values(["id", "idx"])
    scores = {mode: {i: g[f"p_{mode}"].values for i, g in pr.groupby("id")} for mode in ("l1", "l2")}
    configs = _configs(cfg, has_turn_p)

    def subset(split: str, lang: str | None):
        return [i for i, (s, lg) in meta.items() if s == split and (lang is None or lg == lang)]

    out = {"time_to_ready_ms": cfg.time_to_ready_ms, "endpoint_delay_ms": cfg.endpoint_delay_ms,
           "max_speculations": cfg.max_speculations, "points": {}, "selected": {},
           "ttr_sensitivity": {}, "cost_view": {}}
    sets = {f"{split}_{lang or 'all'}": subset(split, lang)
            for split in ("val", "test") for lang in (None, *LANGS)}

    for mode in ("l1", "l2"):
        pts = []
        for name, ids in sets.items():
            vs = [views[i] for i in ids]
            for fam, prm in configs:
                trig = [_trigger(fam, prm, views[i], scores[mode][i]) for i in ids]
                max_spec = None if fam == "every" else cfg.max_speculations
                r = evaluate(vs, trig, mode, cfg.time_to_ready_ms, max_spec, cfg.endpoint_delay_ms)
                pts.append({"set": name, "family": fam, "param": prm, **r})
        out["points"][mode] = pts
        df = pd.DataFrame(pts)

        sel = {}
        for budget in cfg.waste_budgets:
            sel[str(budget)] = {}
            tunings = [("pause", "val_en", "pause (tuned on English val)"),
                       ("pause", "val_all", "pause (tuned on all-language val)"),
                       ("stable", "val_all", "stable (tuned on all-language val)"),
                       ("jaldi", "val_all", "Jaldi (tuned on all-language val)")]
            if has_turn_p:
                tunings.append(("turn_detector", "val_all", "turn detector (tuned on all-language val)"))
            for fam, tune_set, label in tunings:
                cand = df[(df.family == fam) & (df.set == tune_set) & (df.wasted_per_turn <= budget)]
                if cand.empty:
                    sel[str(budget)][label] = {"param": None, "note": "no setting meets the budget on val"}
                    continue
                prm = cand.sort_values(["mean_saved_ms", "wasted_per_turn"], ascending=[False, True]).iloc[0]["param"]
                res = {}
                for lang in ("all", *LANGS):
                    row = df[(df.family == fam) & (df.set == f"test_{lang}") & (df.param == prm)].iloc[0]
                    res[lang] = {k: float(row[k]) for k in ("mean_saved_ms", "wasted_per_turn",
                                                           "commit_rate", "wrong_commit_rate",
                                                           "latency_p50_ms", "latency_p90_ms")}
                sel[str(budget)][label] = {"param": float(prm), "test": res}
        out["selected"][mode] = sel

        # time-to-ready sensitivity for the selected configs at the tightest budget
        b0 = str(cfg.waste_budgets[0])
        sens = {}
        for label, s in sel[b0].items():
            if s.get("param") is None:
                continue
            fam = label.split(" ")[0].lower()
            fam = "turn_detector" if fam == "turn" else fam
            sens[label] = {}
            for ttr in cfg.time_to_ready_sensitivity:
                sens[label][str(ttr)] = {}
                for lang in LANGS:
                    ids = sets[f"test_{lang}"]
                    trig = [_trigger(fam, s["param"], views[i], scores[mode][i]) for i in ids]
                    r = evaluate([views[i] for i in ids], trig, mode, ttr, cfg.max_speculations,
                                 cfg.endpoint_delay_ms)
                    sens[label][str(ttr)][lang] = {"mean_saved_ms": r["mean_saved_ms"],
                                                   "wasted_per_turn": r["wasted_per_turn"]}
        out["ttr_sensitivity"][mode] = sens

        # cost view: cost per turn = c * wasted calls - ms saved  (c in ms per wasted call)
        cv = {}
        for c in cfg.cost_per_wasted_call_ms:
            cv[str(c)] = {}
            for lang in LANGS:
                rows = df[(df.set == f"test_{lang}") & df.family.isin(["jaldi", "pause"])].copy()
                rows["cost"] = c * rows.wasted_per_turn - rows.mean_saved_ms
                curve = rows[rows.family == "jaldi"].sort_values("param")
                best_j = curve.loc[curve.cost.idxmin()]
                pause_rows = rows[rows.family == "pause"]
                best_p = pause_rows.loc[pause_rows.cost.idxmin()]
                cv[str(c)][lang] = {
                    "jaldi_curve": [{"tau": float(r.param), "cost": float(r.cost)} for r in curve.itertuples()],
                    "best_tau": float(best_j.param), "best_jaldi_cost": float(best_j.cost),
                    "best_pause_x": float(best_p.param), "best_pause_cost": float(best_p.cost),
                    "note": "best settings chosen on test (oracle), for illustration only",
                }
        out["cost_view"][mode] = cv
    return out
