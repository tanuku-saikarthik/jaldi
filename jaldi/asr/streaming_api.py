"""Fidelity check: how do simulated partials compare to a real streaming recognizer?

`revision_stats` works on any partial records (sim or real) and is used for the comparison.
`deepgram_partials` streams a wav to Deepgram with interim results (needs DEEPGRAM_API_KEY and
the `websockets` package). It is written against Deepgram's documented live API but has NOT been
run in this repo yet; treat its output with care until verified.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..text import tokens


def revision_stats(records: list[dict]) -> dict:
    """Per-partial revision rate (earlier words changed, not just appended) and words changed."""
    revs, changed, n = 0, [], 0
    for r in records:
        prev = []
        for p in r["partials"]:
            cur = tokens(p["text"])
            if prev and cur[: len(prev)] != prev:
                revs += 1
                k = 0
                while k < min(len(prev), len(cur)) and prev[k] == cur[k]:
                    k += 1
                changed.append(len(prev) - k)
            prev = cur
            n += 1
    return {"partials": n, "revision_rate": revs / max(1, n),
            "mean_words_rewritten_per_revision": float(np.mean(changed)) if changed else 0.0}


async def deepgram_partials(wav_path: str, language: str, step_ms: int = 100,
                            model: str = "nova-3") -> list[dict]:
    import asyncio  # noqa: PLC0415

    import soundfile as sf  # noqa: PLC0415
    import websockets  # noqa: PLC0415

    key = os.environ["DEEPGRAM_API_KEY"]
    wav, sr = sf.read(wav_path, dtype="int16")
    url = (f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={sr}&channels=1"
           f"&interim_results=true&language={language}&model={model}")
    finals, partials = [], []
    t0 = None
    async with websockets.connect(url, additional_headers={"Authorization": f"Token {key}"}) as ws:
        async def sender():
            nonlocal t0
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            step = int(sr * step_ms / 1000)
            for s in range(0, len(wav), step):
                await ws.send(wav[s:s + step].tobytes())
                await asyncio.sleep(step_ms / 1000)  # real-time pacing
            await ws.send(json.dumps({"type": "CloseStream"}))

        async def receiver():
            loop = asyncio.get_event_loop()
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") != "Results":
                    continue
                alt = m["channel"]["alternatives"][0]["transcript"]
                text = " ".join(finals + ([alt] if alt else []))
                if m.get("is_final") and alt:
                    finals.append(alt)
                partials.append({"t_ms": (loop.time() - t0) * 1000, "text": text})

        await asyncio.gather(sender(), receiver())
    return partials
