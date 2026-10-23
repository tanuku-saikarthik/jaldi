"""Measure time-to-ready on GPU: LLM time to first ~sentence + an assumed TTS first-chunk latency.

ttr = time to generate `first_chunk_tokens` tokens for a short agent reply + tts_first_byte_ms.
Writes data/ttr.json; `jaldi simulate` uses its p50 when the file exists. Not run on CPU.
"""

from __future__ import annotations

import time

import numpy as np

PROMPTS = [
    "Caller: mujhe kal chaar logon ke liye table book karni hai. Reply in one short sentence.",
    "Caller: I want to cancel my cab to the airport. Reply in one short sentence.",
    "Caller: मुझे परसों का अपॉइंटमेंट कैंसल करना है. Reply in one short sentence.",
    "Caller: please track my headphones order. Reply in one short sentence.",
]


def measure_ttr(model: str, first_chunk_tokens: int = 12, tts_first_byte_ms: float = 150.0,
                repeats: int = 5) -> dict:
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(model=model, max_model_len=2048, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True) for p in PROMPTS]
    llm.generate(prompts[:1], SamplingParams(max_tokens=4))  # warm-up
    ttft, chunk = [], []
    for _ in range(repeats):
        for p in prompts:
            t0 = time.perf_counter()
            llm.generate([p], SamplingParams(max_tokens=1, temperature=0))
            ttft.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            llm.generate([p], SamplingParams(max_tokens=first_chunk_tokens, temperature=0))
            chunk.append((time.perf_counter() - t0) * 1000)
    ttr = np.array(chunk) + tts_first_byte_ms
    return {"model": model, "ttft_ms_p50": float(np.median(ttft)), "first_chunk_ms_p50": float(np.median(chunk)),
            "tts_first_byte_ms_assumed": tts_first_byte_ms, "ttr_ms_p50": float(np.median(ttr)),
            "ttr_ms_p90": float(np.percentile(ttr, 90)),
            "note": "single-request offline vLLM on the labelling GPU; production latency will differ"}
