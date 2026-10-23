"""Stage 3 (GPU): simulated streaming partials from real audio.

Every `partial_step_ms` the ASR transcribes the audio prefix [0, t]. This mimics a streaming
recognizer that re-decodes its buffer; it is not identical to a true streaming model (see
streaming_api.py for the fidelity check). Not exercised in the CPU smoke run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..audio.prosody import end_prosody, silence_and_speech_ms
from ..config import Config
from ..io import read_jsonl, write_jsonl

SR = 16000
WHISPER_LANG = {"en": "en", "hi": "hi", "hinglish": "hi"}  # revisit after the script check
HALLUCINATIONS = {"thank you.", "thanks for watching!", "धन्यवाद", "subtitles by the amara.org community"}


class WhisperASR:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "cuda"):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.dtype).to(device)
        self.proc = AutoProcessor.from_pretrained(model_id)

    def __call__(self, wavs: list[np.ndarray], lang: str) -> list[tuple[str, float]]:
        torch = self.torch
        feats = self.proc(wavs, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.dtype)
        with torch.no_grad():
            gen = self.model.generate(feats, language=lang, task="transcribe", max_new_tokens=96,
                                      return_dict_in_generate=True, output_scores=True)
        texts = self.proc.batch_decode(gen.sequences, skip_special_tokens=True)
        try:
            tr = self.model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
            lp = tr.float().cpu().numpy()
            confs = [float(np.exp(np.mean(row[np.isfinite(row) & (row < 0)]))) if np.any(row < 0) else math.nan
                     for row in lp]
        except Exception:  # noqa: BLE001
            confs = [math.nan] * len(texts)
        out = []
        for txt, c in zip(texts, confs):
            txt = txt.strip()
            if txt.lower() in HALLUCINATIONS:
                txt = ""
            out.append((txt, c))
        return out


class IndicConformerASR:
    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual", device: str = "cuda",
                 decoding: str = "ctc"):
        import torch  # noqa: PLC0415
        from transformers import AutoModel  # noqa: PLC0415

        self.torch = torch
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        try:
            self.model = self.model.to(device)
        except Exception:  # noqa: BLE001  (ONNX-backed remote code may manage its own device)
            pass
        self.decoding = decoding

    def __call__(self, wavs: list[np.ndarray], lang: str) -> list[tuple[str, float]]:
        out = []
        for w in wavs:
            t = self.torch.from_numpy(w).unsqueeze(0)
            txt = self.model(t, "hi" if lang in ("hi", "hinglish") else lang, self.decoding)
            out.append((str(txt).strip(), math.nan))
        return out


def audio_partials(corpus_path: str, timing_path: str, wav_dir: str, out_path: str, cfg: Config,
                   asr_by_lang: dict[str, str], use_turn_detector: bool = True,
                   batch_size: int = 32, limit: int | None = None) -> dict:
    import soundfile as sf  # noqa: PLC0415

    corpus = {r["id"]: r for r in read_jsonl(corpus_path)}
    timing = read_jsonl(timing_path)
    if limit:
        timing = timing[:limit]
    engines = {}
    for name in set(asr_by_lang.values()):
        engines[name] = WhisperASR() if name == "whisper" else IndicConformerASR()
    td = None
    if use_turn_detector:
        from ..features.turn_detector import TurnDetector  # noqa: PLC0415

        td = TurnDetector()

    records = []
    for tm in timing:
        t = corpus[tm["id"]]
        wav, sr = sf.read(Path(wav_dir) / f"{t['id']}.wav", dtype="float32")
        assert sr == SR
        eot = tm["eot_ms"]
        times = []
        x = cfg.partial_step_ms
        while x < eot:
            times.append(float(x))
            x += cfg.partial_step_ms
        times.append(float(eot))
        prefixes = [wav[: int(tt / 1000 * SR)] for tt in times]
        info = [silence_and_speech_ms(p, SR) for p in prefixes]
        keep = [i for i, (_, sp) in enumerate(info) if sp >= 300]
        name = asr_by_lang[t["lang"]]
        texts = {}
        for s in range(0, len(keep), batch_size):
            ks = keep[s:s + batch_size]
            res = engines[name]([prefixes[i] for i in ks], WHISPER_LANG[t["lang"]])
            for i, r in zip(ks, res):
                texts[i] = r
        partials = []
        for i in keep:
            txt, conf = texts[i]
            if not txt:
                continue
            f0s, edrop = end_prosody(prefixes[i], SR)
            partials.append({
                "t_ms": times[i], "text": txt, "asr_conf": conf,
                "silence_ms": info[i][0], "speech_ms": info[i][1],
                "f0_slope": f0s, "energy_drop": edrop,
                "turn_p": td(prefixes[i]) if td else math.nan,
            })
        if not partials:
            continue
        full_text = engines[name]([wav], WHISPER_LANG[t["lang"]])[0][0]
        records.append({
            "id": t["id"], "mode": "audio", "asr": name, "eot_ms": eot,
            "speech_end_ms": tm["speech_end_ms"], "final_text": partials[-1]["text"],
            "full_clip_text": full_text, "partials": partials,
        })
    write_jsonl(out_path, records)
    from ..text import normalize  # noqa: PLC0415

    agree = np.mean([normalize(r["final_text"]) == normalize(r["full_clip_text"]) for r in records]) if records else 0
    return {"turns": len(records), "partials_per_turn": float(np.mean([len(r["partials"]) for r in records]))
            if records else 0, "final_partial_equals_full_clip": float(agree)}


def script_check(corpus_path: str, seg_out: str, n: int = 10) -> dict:
    """Hinglish: synthesize n turns in romanized vs mixed script, compare action round-trip after ASR."""
    import random  # noqa: PLC0415

    import soundfile as sf  # noqa: PLC0415

    from ..audio.assemble import assemble_turn  # noqa: PLC0415
    from ..audio.tts import synthesize  # noqa: PLC0415
    from ..labels.rule_extractor import extract, same_action  # noqa: PLC0415

    cfg = Config()
    turns = [t for t in read_jsonl(corpus_path) if t["lang"] == "hinglish"][:n]
    ids = {t["id"] for t in turns}
    asr = WhisperASR()
    res = {}
    for script in ("roman", "mixed"):
        d = Path(seg_out) / script
        synthesize(corpus_path, str(d), hinglish_script=script, ids=ids)
        ok = 0
        for t in turns:
            segs, srs = zip(*(sf.read(d / f"{t['id']}_{k:02d}.wav", dtype="float32")
                              for k in range(len(t["segments"]))))
            wav, _ = assemble_turn(t, list(segs), srs[0], cfg, random.Random(0), False)
            txt = asr([wav], "hi")[0][0]
            ok += same_action(extract(txt, t["scenario"]), t["gold_action"])
        res[script] = {"action_round_trip": ok / max(1, len(turns))}
    res["pick"] = max(("roman", "mixed"), key=lambda s: res[s]["action_round_trip"])
    return res
