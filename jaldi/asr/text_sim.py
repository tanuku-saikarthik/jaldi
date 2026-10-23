"""CPU smoke mode: simulate streaming partials from text + a timing model.

This is NOT ASR. It exists so the whole pipeline (labels, features, model, policy simulation,
report) runs end to end on a laptop. Real partials come from `jaldi.asr.partials` on GPU.
What it models: words appear `lag` ms after they end; an in-progress word sometimes shows up early
as a prefix; the last word is sometimes wrong for one partial and then fixed (a revision).
What it does not model: real ASR errors, script choices, code-mix confusions, hallucinations.
"""

from __future__ import annotations

import math
import random

from ..audio.timing import word_timeline
from ..config import Config

NAN = float("nan")


def _corrupt(word: str) -> str:
    core = word.strip(",.?")
    return core[:-1] if len(core) > 2 else core + core[-1:]


def simulate_turn(turn: dict, cfg: Config) -> dict:
    rng = random.Random(f"{cfg.seed}-{turn['id']}")
    words = word_timeline(turn, cfg, rng)
    speech_start, speech_end = words[0]["start"], words[-1]["end"]
    eot = speech_end + cfg.endpoint_delay_ms
    lag = cfg.sim_asr_lag_ms
    final_text = " ".join(w["word"] for w in words)

    partials = []
    t = cfg.partial_step_ms
    while True:
        t = min(t, eot)
        is_final = t >= eot
        vis = [w["word"] for w in words if w["end"] + lag <= t]
        if not is_final:
            in_prog = [w for w in words if w["start"] + lag <= t < w["end"] + lag]
            if in_prog and rng.random() < cfg.sim_p_early_word:
                w = in_prog[0]
                frac = (t - w["start"] - lag) / (w["end"] - w["start"])
                core = w["word"].strip(",.?")
                vis = vis + [core[: max(1, math.ceil(frac * len(core)))]]
            elif vis and rng.random() < cfg.sim_p_revision:
                vis = vis[:-1] + [_corrupt(vis[-1])]
        text = final_text if is_final else " ".join(vis)
        speaking = any(w["start"] <= t < w["end"] for w in words)
        last_end = max((w["end"] for w in words if w["end"] <= t), default=None)
        silence = 0.0 if speaking or last_end is None else t - last_end
        if text:
            partials.append({
                "t_ms": round(t, 1),
                "text": text,
                "asr_conf": NAN,
                "silence_ms": round(silence, 1),
                "speech_ms": round(max(0.0, t - speech_start), 1),
                "f0_slope": NAN,
                "energy_drop": NAN,
                "turn_p": NAN,
            })
        if is_final:
            break
        t += cfg.partial_step_ms

    return {
        "id": turn["id"],
        "mode": "text_sim",
        "eot_ms": round(eot, 1),
        "speech_end_ms": round(speech_end, 1),
        "final_text": final_text,
        "partials": partials,
    }
