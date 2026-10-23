"""Stage 2: join per-chunk TTS audio with sampled pauses; optional light telephony degradation.

Pauses use the same distribution as the text simulator (jaldi.audio.timing), identical across
languages. Output: 16 kHz mono turn audio + exact segment timestamps.
"""

from __future__ import annotations

import random
from math import gcd
from pathlib import Path

import numpy as np

from ..config import Config
from ..io import read_jsonl, write_jsonl
from .timing import chunk_pauses

SR = 16000


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from scipy.signal import resample_poly  # noqa: PLC0415

    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def trim(x: np.ndarray, sr: int, thresh_db: float = -40.0, margin_ms: float = 30.0) -> np.ndarray:
    """Trim leading/trailing low-energy audio so inserted pauses are the only pauses."""
    if len(x) == 0:
        return x
    frame = int(0.01 * sr)
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (np.max(rms) + 1e-12))
    voiced = np.where(db > thresh_db)[0]
    if len(voiced) == 0:
        return x[:0]
    m = int(margin_ms / 1000 * sr)
    return x[max(0, voiced[0] * frame - m): min(len(x), (voiced[-1] + 1) * frame + m)]


def telephony(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """8 kHz band-limit, 8-bit mu-law companding, mild noise (SNR 20-30 dB), back to 16 kHz."""
    y = resample(x, SR, 8000)
    mu = 255.0
    y = np.clip(y, -1, 1)
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * 255) / 255 * 2 - 1
    y = np.sign(c) * (np.power(1 + mu, np.abs(c)) - 1) / mu
    snr = rng.uniform(20, 30)
    p = np.mean(y ** 2) + 1e-12
    nprng = np.random.default_rng(rng.randrange(2 ** 31))
    y = y + nprng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(y))
    return resample(y.astype(np.float32), 8000, SR)


def assemble_turn(turn: dict, segs: list[np.ndarray], sr_in: int, cfg: Config,
                  rng: random.Random, degrade: bool) -> tuple[np.ndarray, dict]:
    pauses = chunk_pauses(turn, cfg, rng)
    lead = np.zeros(int(0.3 * SR), np.float32)
    parts, seg_times, t = [lead], [], 300.0
    for seg, pause in zip(segs, pauses):
        if pause:
            parts.append(np.zeros(int(pause / 1000 * SR), np.float32))
            t += pause
        a = trim(resample(seg, sr_in, SR), SR)
        dur = len(a) / SR * 1000
        seg_times.append({"start_ms": round(t, 1), "end_ms": round(t + dur, 1)})
        parts.append(a)
        t += dur
    tail = np.zeros(int((cfg.endpoint_delay_ms + 300) / 1000 * SR), np.float32)
    wav = np.concatenate(parts + [tail])
    peak = np.max(np.abs(wav)) + 1e-9
    wav = (0.9 * wav / peak).astype(np.float32)
    if degrade:
        wav = telephony(wav, rng)
    speech_end = seg_times[-1]["end_ms"]
    return wav, {"id": turn["id"], "segments": seg_times, "speech_start_ms": seg_times[0]["start_ms"],
                 "speech_end_ms": speech_end, "eot_ms": round(speech_end + cfg.endpoint_delay_ms, 1),
                 "telephony": degrade, "duration_ms": round(len(wav) / SR * 1000, 1)}


def assemble_all(corpus_path: str, seg_dir: str, out_dir: str, timing_path: str, cfg: Config,
                 p_telephony: float = 0.5) -> dict:
    import soundfile as sf  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], 0
    for t in read_jsonl(corpus_path):
        paths = [Path(seg_dir) / f"{t['id']}_{k:02d}.wav" for k in range(len(t["segments"]))]
        if not all(p.exists() for p in paths):
            missing += 1
            continue
        segs, srs = zip(*(sf.read(p, dtype="float32") for p in paths))
        rng = random.Random(f"{cfg.seed}-{t['id']}-asm")
        wav, timing = assemble_turn(t, list(segs), srs[0], cfg, rng, rng.random() < p_telephony)
        sf.write(out / f"{t['id']}.wav", wav, SR)
        rows.append(timing)
    write_jsonl(timing_path, rows)
    return {"assembled": len(rows), "missing_segments": missing}
