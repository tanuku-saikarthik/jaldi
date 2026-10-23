"""Cheap causal prosody: energy VAD, F0 slope and energy drop at the end of the audio prefix.

Pure numpy, runs on CPU. Every function only looks at the samples it is given (the prefix).
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 20


def frame_db(x: np.ndarray, sr: int, frame_ms: int = FRAME_MS) -> np.ndarray:
    f = int(sr * frame_ms / 1000)
    n = len(x) // f
    if n == 0:
        return np.array([])
    rms = np.sqrt(np.mean(x[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def vad(x: np.ndarray, sr: int, abs_floor_db: float = -45.0, rel_db: float = 30.0,
        hangover_frames: int = 3) -> np.ndarray:
    """Boolean speech mask per frame. Threshold = max(abs floor, loudest frame so far - rel_db)."""
    db = frame_db(x, sr)
    if len(db) == 0:
        return np.zeros(0, bool)
    peak_so_far = np.maximum.accumulate(db)
    speech = db > np.maximum(abs_floor_db, peak_so_far - rel_db)
    out = speech.copy()
    run = 0
    for i, s in enumerate(speech):
        run = hangover_frames if s else max(0, run - 1)
        out[i] = s or run > 0
    return out


def silence_and_speech_ms(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(ms since last speech frame, ms since first speech frame) at the end of the prefix."""
    m = vad(x, sr)
    idx = np.where(m)[0]
    t_end = len(m) * FRAME_MS
    if len(idx) == 0:
        return 0.0, 0.0
    return float(t_end - (idx[-1] + 1) * FRAME_MS), float(t_end - idx[0] * FRAME_MS)


def _f0_autocorr(frame: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> float:
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < 1e-3:
        return np.nan
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return np.nan
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < 0.3 * ac[0]:
        return np.nan
    return sr / lag


def end_prosody(x: np.ndarray, sr: int, window_ms: int = 300) -> tuple[float, float]:
    """(F0 slope in semitones/s, energy drop in dB) over the last `window_ms` of SPEECH in the prefix."""
    m = vad(x, sr)
    idx = np.where(m)[0]
    if len(idx) < 5:
        return np.nan, np.nan
    end = (idx[-1] + 1) * int(sr * FRAME_MS / 1000)
    start = max(0, end - int(sr * window_ms / 1000))
    seg = x[start:end]
    f = int(0.04 * sr)
    hop = int(0.02 * sr)
    f0s, ts = [], []
    for s in range(0, max(0, len(seg) - f), hop):
        v = _f0_autocorr(seg[s:s + f], sr)
        if np.isfinite(v):
            f0s.append(12 * np.log2(v / 100.0))
            ts.append(s / sr)
    slope = float(np.polyfit(ts, f0s, 1)[0]) if len(f0s) >= 3 else np.nan
    db = frame_db(seg, sr)
    if len(db) >= 6:
        k = max(1, len(db) // 3)
        drop = float(np.mean(db[:-k]) - np.mean(db[-k:]))
    else:
        drop = np.nan
    return slope, drop
