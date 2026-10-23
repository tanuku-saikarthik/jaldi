# PROGRESS

Last updated: 2026-10-23. Numbers live in `results/*.json` and the README results block (generated
by `python -m jaldi.cli report`). This file tracks status, not results.

## Status by stage

| Stage | What | Status | Gate |
|---|---|---|---|
| 1 | Template corpus, en/hi/hinglish, 6 scenarios, 5 phenomena | ✅ done (CPU) | PASS: 1,800 turns, 0 invalid gold actions; rule extractor recovers gold on 100% of final transcripts |
| 2 | Indic Parler-TTS per chunk + pause assembly + optional telephony | 🟡 code written, **not run** (needs GPU) | — |
| 3 | Partials | ✅ text-sim mode (CPU); 🟡 audio mode (prefix ASR + VAD + prosody + turn detector) written, **not run** | PASS (text-sim): final partial == final transcript, ≥3 partials/turn |
| 4 | L1/L2 labels + M1 analysis | ✅ rule labels (CPU); 🟡 vLLM labeller written, **not run** | PASS: final action == gold on 100% (rule extractor on clean text, so this is trivially true here) |
| 5 | Causal features | ✅ done | PASS: leakage tests (truncation + future edits) green |
| 6 | LightGBM ×5 seeds + isotonic, L1/L2, ablations | ✅ done; 🟡 MuRIL encoder + fusion written, **not run** | PASS on smoke data (see caveat below) |
| 7 | Policy simulation, baselines, budgets, cost view, TTR sensitivity | ✅ done | — |
| 8 | Gradio demo (replay + live CPU) | ✅ replay verified headless; live mode needs `.[demo,gpu]`, not run here | — |
| 9 | README, plots, thread, demo script (all generated from results) | ✅ done for the smoke run | — |
| Stretch | `docs/integration.md` (Bolna hook) | ✅ Bolna read at `7de6108`; pipecat not read | — |

## What the smoke run is and is not

- **It is:** an end-to-end check of the pipeline plus the *shape* of the result on templates. Verb-final
  word order gives Hindi/Hinglish more negation flips AND more headroom. A pause rule is
  word-order-blind; a learned policy can use the headroom.
- **It is not:** evidence about real callers or real ASR. Template text makes the cues nearly
  deterministic, so AUROC (~0.96 L2, ~0.99 L1) is far above what real speech will give. The gate
  passing here means little; the GPU run is the real test.

## Open issues

1. **None of the GPU code has been executed yet**: `audio/tts.py`, `audio/assemble.py` (I/O
   path), `asr/partials.py`, `labels/action_match.label_with_vllm`, `sim/ttr.py`,
   `models/text_encoder.py`, `features/turn_detector.py`, `asr/streaming_api.deepgram_partials`.
   Expect small fixes on the first Colab run (library versions, Parler batch output shape).
2. **Parler speaker names** in `corpus/lexicon.py::SPEAKERS` must be checked against the model card.
3. **Hinglish ASR script**: Whisper is called with `language="hi"` for Hinglish until `script-check` decides.
4. **Per-language waste**: τ is tuned on all-language val, so English waste can exceed the
   budget while Hindi stays well under it. Try per-language τ (the language is known at runtime).
5. **Pause thresholds above the endpointing delay (500 ms)** can only fire on mid-turn pauses, so at
   tight budgets the "best" pause setting saves ~0 ms. The thread and demo script headline the loosest
   budget (≤1.0 wasted call/turn), where the heuristic is competitive.
6. **Turn detector**: the default is the community Smart Turn Hinglish ONNX. Compare against the
   official pipecat smart-turn release.
7. **Real-speech test set** (~50 human-recorded turns) not collected yet.
8. Partial fidelity check vs a real streaming API needs a provider key. Not run.

## Next steps

1. Run `notebooks/gpu_stages.ipynb` Session A, then B. Fix whatever breaks and log it here.
2. `jaldi report` → README switches from "Smoke run" automatically (mode=audio).
3. Record the human mini test set and report it separately.
4. Per-language τ; feature-group ablation on real audio (prosody and turn-detector features are NaN in the smoke run).
