"""Pause sampling shared by the real audio assembler and the text simulator.

Both use the same pause distribution for every language, so pauses carry no language signal.
"""

from __future__ import annotations

import hashlib
import random

from ..config import Config


def speaker_rate(speaker: str) -> float:
    """Deterministic speaking-rate factor per speaker (1.0 = average)."""
    h = int(hashlib.md5(speaker.encode()).hexdigest()[:8], 16)
    return 0.85 + 0.35 * (h % 1000) / 1000


def sample_pause_ms(rng: random.Random, cfg: Config, forced: str | None) -> float:
    kind = forced or rng.choices(list(cfg.pause_probs), list(cfg.pause_probs.values()))[0]
    lo, hi = {"none": cfg.none_pause_ms, "short": cfg.short_pause_ms, "long": cfg.long_pause_ms}[kind]
    return rng.uniform(lo, hi)


def chunk_pauses(turn: dict, cfg: Config, rng: random.Random) -> list[float]:
    """Pause (ms) before each chunk; the first is 0."""
    forced = turn.get("forced_pause_before", {})
    return [0.0] + [sample_pause_ms(rng, cfg, forced.get(str(i)))
                    for i in range(1, len(turn["segments"]))]


def word_timeline(turn: dict, cfg: Config, rng: random.Random) -> list[dict]:
    """Text-sim word timings: [{word, start, end}] in ms from turn start (speech starts at 300 ms)."""
    rate = speaker_rate(turn.get("speaker", "x")) * rng.uniform(0.92, 1.08)
    pauses = chunk_pauses(turn, cfg, rng)
    t, words = 300.0, []
    for seg, pause in zip(turn["segments"], pauses):
        t += pause
        for j, w in enumerate(seg.split()):
            if j:
                t += rng.uniform(0, 60)
            n = len(w.strip(",.?"))
            dur = rate * (120 + 42 * n)
            words.append({"word": w, "start": t, "end": t + dur})
            t += dur
    return words
