"""Jaldi demo.

Replay (default, no GPU, no model needed): curated test turns from results/demo_examples.json,
showing partials as they arrive, Jaldi's P(safe), where the pause heuristic and Jaldi fire, and the
outcome (ms saved / calls wasted) for each.

Live (optional, CPU): record or upload a short clip. A small Whisper ("CPU demo model") re-decodes
the prefix every 200 ms and the trained Jaldi model scores each partial. Needs `pip install -e
".[demo,gpu]"` and a trained model in data/models (run `python -m jaldi.cli all` first). The smoke
model was trained on template text, so live scores on real speech are illustrative only.

Run: python app/demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jaldi.eval.plots import FAM_STYLE, GRID, INK, INK2, SURFACE, _style  # noqa: E402

EXAMPLES = json.loads((ROOT / "results" / "demo_examples.json").read_text(encoding="utf-8"))
NAME = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}


def _label(e: dict) -> str:
    return f"{NAME[e['lang']]} · {e['phenomenon']} · {e['final_text'][:60]}"


def timeline(partials: list[dict], eot: float, tau: float, title: str, pause_x: float | None = None):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ts = [p["t_ms"] for p in partials]
    ps = [p["p_safe"] for p in partials]
    jcol = FAM_STYLE["jaldi"][0]
    hcol = FAM_STYLE["pause"][0]
    ax.plot(ts, ps, color=jcol, lw=2, marker="o", ms=5, label="Jaldi P(safe)")
    ax.axhline(tau, color=jcol, lw=1, ls=":")
    ax.annotate(f"τ = {tau:g}", (ts[0], tau), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
    if "l2" in partials[0]:
        for t, p in zip(ts, partials):
            if p["l2"]:
                ax.axvspan(t - 100, t + 100, color=GRID, alpha=0.5, lw=0)
    first_j = next((p["t_ms"] for p in partials if p.get("jaldi_trigger")), None)
    first_h = next((p["t_ms"] for p in partials if p.get("pause_trigger")), None)
    if first_h is not None:
        ax.axvline(first_h, color=hcol, lw=2, ls="--", label=f"pause ≥ {pause_x:.0f} ms first fires")
    if first_j is not None:
        ax.axvline(first_j, color=jcol, lw=2, label="Jaldi first fires")
    ax.axvline(eot, color=INK, lw=1.5, label="end of turn")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("ms since turn start  (shaded = action already final)", color=INK2)
    ax.set_ylabel("P(safe)", color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def replay(choice: str):
    e = next(x for x in EXAMPLES if _label(x) == choice)
    fig = timeline(e["partials"], e["eot_ms"], e["tau"], e["final_text"], e["pause_x_ms"])
    rows = [[f"{p['t_ms']:.0f}", p["text"], f"{p['silence_ms']:.0f}", f"{p['p_safe']:.2f}",
             "✓" if p["l2"] else "", "●" if p["pause_trigger"] else "", "●" if p["jaldi_trigger"] else ""]
            for p in e["partials"]]
    h, j = e["heuristic"], e["jaldi"]
    summary = (f"**Gold action:** `{json.dumps(e['gold_action'], ensure_ascii=False)}`\n\n"
               f"**Pause ≥ {e['pause_x_ms']:.0f} ms:** saved {h['saved']:.0f} ms, wasted {h['wasted']} call(s)  \n"
               f"**Jaldi (τ={e['tau']:g}):** saved {j['saved']:.0f} ms, wasted {j['wasted']} call(s)  \n"
               f"Time-to-ready assumed {e['time_to_ready_ms']:.0f} ms. Smoke-run example (simulated partials).")
    return fig, rows, summary


_LIVE = {}


def live(audio):
    if audio is None:
        return None, "Record or upload a clip first."
    try:
        import numpy as np
        import torch  # noqa: F401
        from transformers import pipeline

        from jaldi.audio.assemble import resample
        from jaldi.audio.prosody import end_prosody, silence_and_speech_ms
        from jaldi.models.scorer import JaldiScorer
    except ImportError as ex:
        return None, f"Live mode needs extra packages: {ex}. Install with `pip install -e \".[demo,gpu]\"`."
    if "asr" not in _LIVE:
        _LIVE["asr"] = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
        _LIVE["scorer"] = JaldiScorer(ROOT / "data" / "models", "l2")
    sr, wav = audio
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = wav / (np.abs(wav).max() + 1e-9)
    wav = resample(wav, sr, 16000)
    partials, t = [], 200
    while t <= len(wav) / 16:
        pre = wav[: t * 16]
        sil, sp = silence_and_speech_ms(pre, 16000)
        if sp >= 300:
            txt = _LIVE["asr"]({"raw": pre, "sampling_rate": 16000})["text"].strip()
            if txt:
                f0, ed = end_prosody(pre, 16000)
                partials.append({"t_ms": float(t), "text": txt, "silence_ms": sil, "speech_ms": sp,
                                 "asr_conf": float("nan"), "f0_slope": f0, "energy_drop": ed,
                                 "turn_p": float("nan")})
        t += 200
    if not partials:
        return None, "No speech detected."
    lang = "hi" if any("ऀ" <= ch <= "ॿ" for ch in partials[-1]["text"]) else "en"
    scores = _LIVE["scorer"].score(partials, lang)
    for p, s in zip(partials, scores):
        p["p_safe"] = s
    fig = timeline(partials, len(wav) / 16, 0.7, "Live (CPU demo model: whisper-tiny)")
    lines = "\n".join(f"- {p['t_ms']:.0f} ms · P(safe) {p['p_safe']:.2f} · {p['text']}" for p in partials)
    return fig, lines


def main():
    import gradio as gr

    with gr.Blocks(title="Jaldi") as demo:
        gr.Markdown("# Jaldi — when can a voice agent start answering?\n"
                    "Replay curated test turns: where does a pause heuristic fire, where does Jaldi fire, "
                    "and what does each cost?")
        with gr.Tab("Replay"):
            choice = gr.Dropdown([_label(e) for e in EXAMPLES], value=_label(EXAMPLES[0]), label="Example")
            plot = gr.Plot()
            summary = gr.Markdown()
            table = gr.Dataframe(headers=["t (ms)", "partial", "silence (ms)", "P(safe)", "action final",
                                          "pause fires", "Jaldi fires"], wrap=True)
            choice.change(replay, choice, [plot, table, summary])
            demo.load(replay, choice, [plot, table, summary])
        with gr.Tab("Live (CPU demo model)"):
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy")
            btn = gr.Button("Score")
            lplot = gr.Plot()
            ltext = gr.Markdown()
            btn.click(live, audio, [lplot, ltext])
    demo.launch()


if __name__ == "__main__":
    main()
