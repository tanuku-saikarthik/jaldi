"""Pause / pace features and prosody pass-through.

`silence_ms`, `speech_ms` come from the partial record (energy VAD in real mode).
`f0_slope`, `energy_drop` are computed on GPU/CPU from audio by `jaldi.audio.prosody`;
`turn_p` by `jaldi.features.turn_detector`. In text-sim mode they are NaN (LightGBM handles NaN).
"""

from __future__ import annotations

import math

ACOUSTIC_FEATURES = ["silence_ms", "speech_ms", "speech_rate_wps", "f0_slope", "energy_drop"]
TURN_FEATURES = ["turn_p"]


def acoustic_features(p: dict, n_words: int) -> dict:
    sp = p.get("speech_ms", 0.0) or 0.0
    return {
        "silence_ms": p.get("silence_ms", math.nan),
        "speech_ms": sp,
        "speech_rate_wps": (n_words / (sp / 1000.0)) if sp > 0 else math.nan,
        "f0_slope": p.get("f0_slope", math.nan),
        "energy_drop": p.get("energy_drop", math.nan),
    }


def turn_features(p: dict) -> dict:
    return {"turn_p": p.get("turn_p", math.nan)}
