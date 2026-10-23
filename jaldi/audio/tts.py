"""Stage 2 (GPU): synthesize every chunk of every turn with Indic Parler-TTS.

Each chunk is synthesized separately so pauses can be inserted between chunks afterwards
(see assemble.py). Speaker style (pace/pitch) is fixed within a turn and varies across turns.
Not exercised in the CPU smoke run.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from ..corpus.generate import hinglish_mixed_script
from ..io import read_jsonl, write_jsonl

PACES = ["slightly slow", "moderate", "slightly fast"]
PITCHES = ["low", "moderate", "slightly high"]


def caption(speaker: str, rng: random.Random) -> str:
    return (f"{speaker}'s voice is clear and expressive, speaking at a {rng.choice(PACES)} pace "
            f"with a {rng.choice(PITCHES)} pitch, in a very close recording with no background noise.")


def segment_text(turn: dict, seg: str, hinglish_script: str) -> str:
    if turn["lang"] == "hinglish" and hinglish_script == "mixed":
        return hinglish_mixed_script(seg)
    return seg


class ParlerTTS:
    def __init__(self, model_id: str = "ai4bharat/indic-parler-tts", device: str = "cuda"):
        import torch  # noqa: PLC0415
        from parler_tts import ParlerTTSForConditionalGeneration  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = self.model.config.sampling_rate

    def batch(self, texts: list[str], descriptions: list[str], seed: int):
        """Returns a list of float32 numpy arrays at self.sr."""
        torch = self.torch
        torch.manual_seed(seed)
        d = self.desc_tok(descriptions, return_tensors="pt", padding=True).to(self.device)
        p = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=d.input_ids, attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                do_sample=True, return_dict_in_generate=True,
            )
        seqs = gen.sequences.float().cpu().numpy()
        lens = getattr(gen, "audios_length", None)
        out = []
        for i in range(len(texts)):
            a = seqs[i]
            if lens is not None:
                a = a[: int(lens[i])]
            out.append(a.astype("float32"))
        return out


def synthesize(corpus_path: str, out_dir: str, hinglish_script: str = "roman", limit: int | None = None,
               batch_size: int = 16, seed: int = 13, ids: set | None = None) -> dict:
    import soundfile as sf  # noqa: PLC0415

    turns = read_jsonl(corpus_path)
    if ids is not None:
        turns = [t for t in turns if t["id"] in ids]
    if limit:
        turns = turns[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tts = ParlerTTS()
    jobs = []
    for t in turns:
        rng = random.Random(f"{seed}-{t['id']}")
        cap = caption(t["speaker"], rng)
        for k, seg in enumerate(t["segments"]):
            path = out / f"{t['id']}_{k:02d}.wav"
            if not path.exists():
                jobs.append((path, segment_text(t, seg, hinglish_script), cap))
    t0 = time.time()
    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s:s + batch_size]
        audios = tts.batch([j[1] for j in chunk], [j[2] for j in chunk], seed + s)
        for (path, _, _), a in zip(chunk, audios):
            sf.write(path, a, tts.sr)
    dt = time.time() - t0
    manifest = [{"id": t["id"], "n_segments": len(t["segments"]), "sr": tts.sr} for t in turns]
    write_jsonl(out / "manifest.jsonl", manifest)
    return {"turns": len(turns), "segments": len(jobs), "seconds": round(dt, 1),
            "segments_per_second": round(len(jobs) / dt, 3) if dt else None,
            "hinglish_script": hinglish_script}
