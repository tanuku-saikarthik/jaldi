"""Stage 7: replay turns in time and score a speculation policy.

A policy is a per-partial trigger. The engine:
  * starts a speculative LLM call at a triggering partial (at most `max_spec` per turn)
  * discards it (waste) as soon as a later partial changes the transcript (L1) or action (L2)
  * at end of turn, commits it if still valid; saved = min(time_to_ready, eot - start)
The last partial is emitted at end of turn and equals the final transcript, so the commit check
is the same comparison against the final transcript / final action.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..labels import rule_extractor as R
from ..text import normalize


def action_key(a: dict) -> str:
    return json.dumps(a, sort_keys=True, ensure_ascii=False)


@dataclass
class TurnView:
    """Arrays a policy may look at (all causal) plus what the scorer needs."""
    id: str
    lang: str
    t: np.ndarray
    keys_l1: list
    keys_l2: list
    silence: np.ndarray
    stable: np.ndarray
    turn_p: np.ndarray
    eot: float
    speech_end: float
    final_key_l1: str
    final_key_l2: str
    final_matches_gold: bool


def make_view(turn: dict) -> TurnView:
    ps = turn["partials"]
    k1 = [normalize(p["text"]) for p in ps]
    stable, run = [], 0
    for i, k in enumerate(k1):
        run = run + 1 if i and k == k1[i - 1] else 0
        stable.append(run)
    return TurnView(
        id=turn["id"], lang=turn["lang"],
        t=np.array([p["t_ms"] for p in ps], float),
        keys_l1=k1,
        keys_l2=[action_key(p["action"]) for p in ps],
        silence=np.array([p["silence_ms"] for p in ps], float),
        stable=np.array(stable, float),
        turn_p=np.array([p.get("turn_p", np.nan) for p in ps], float),
        eot=float(turn["eot_ms"]), speech_end=float(turn["speech_end_ms"]),
        final_key_l1=normalize(turn["final_text"]),
        final_key_l2=action_key(turn["final_action"]),
        final_matches_gold=bool(turn["final_matches_gold"]),
    )


def run_turn(v: TurnView, trigger: np.ndarray, mode: str, ttr: float, max_spec: int | None) -> dict:
    keys = v.keys_l1 if mode == "l1" else v.keys_l2
    final_key = v.final_key_l1 if mode == "l1" else v.final_key_l2
    spec_i, spec_key, n_spec, wasted = None, None, 0, 0
    for i, k in enumerate(keys):
        if spec_i is not None and k != spec_key:
            wasted += 1
            spec_i = spec_key = None
        if spec_i is None and trigger[i] and (max_spec is None or n_spec < max_spec):
            spec_i, spec_key = i, k
            n_spec += 1
    saved, committed, wrong = 0.0, False, False
    if spec_i is not None:
        if spec_key == final_key:
            committed = True
            saved = float(min(ttr, v.eot - v.t[spec_i]))
            wrong = mode == "l2" and not v.final_matches_gold
        else:
            wasted += 1
    return {"saved": saved, "wasted": wasted, "calls": n_spec, "committed": committed,
            "wrong_commit": wrong}


def evaluate(views: list[TurnView], triggers: list[np.ndarray], mode: str, ttr: float,
             max_spec: int | None, endpoint_delay: float) -> dict:
    rs = [run_turn(v, tr, mode, ttr, max_spec) for v, tr in zip(views, triggers)]
    saved = np.array([r["saved"] for r in rs])
    latency = endpoint_delay + ttr - saved
    return {
        "n_turns": len(rs),
        "mean_saved_ms": float(saved.mean()) if len(rs) else 0.0,
        "commit_rate": float(np.mean([r["committed"] for r in rs])) if rs else 0.0,
        "wasted_per_turn": float(np.mean([r["wasted"] for r in rs])) if rs else 0.0,
        "calls_per_turn": float(np.mean([r["calls"] for r in rs])) if rs else 0.0,
        "wrong_commit_rate": float(np.mean([r["wrong_commit"] for r in rs])) if rs else 0.0,
        "latency_p50_ms": float(np.percentile(latency, 50)) if rs else 0.0,
        "latency_p90_ms": float(np.percentile(latency, 90)) if rs else 0.0,
        "baseline_latency_ms": endpoint_delay + ttr,
    }


def oracle_headroom(turn: dict, mode: str) -> float:
    """ms between the moment the label becomes (and stays) true and end of turn."""
    ps = turn["partials"]
    lab = [p[mode] for p in ps]
    j = len(lab)
    while j > 0 and lab[j - 1]:
        j -= 1
    if j == len(lab):
        return 0.0
    return float(turn["eot_ms"] - ps[j]["t_ms"])


def is_false_actionable(turn: dict) -> bool:
    return any(p["actionable"] and not p["l2"] for p in turn["partials"])


def has_meaning_flip(turn: dict) -> bool:
    """Some partial already looked actionable but contradicts the final action."""
    return any(p["actionable"] and p["contradicts"] for p in turn["partials"])


__all__ = ["R", "TurnView", "make_view", "run_turn", "evaluate", "oracle_headroom",
           "is_false_actionable", "has_meaning_flip"]
