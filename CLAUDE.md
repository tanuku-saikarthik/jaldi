# Jaldi — learning when a voice agent can start answering before you finish

> Drop this file into an empty repo root as `CLAUDE.md`. Then tell Claude Code:
> "Read CLAUDE.md and execute it stage by stage. Update PROGRESS.md after every stage."

## Mission

Every cascaded voice agent (STT → LLM → TTS) pays its LLM + TTS latency **after** the user stops
talking. The known trick is **speculative generation**: start the LLM on a partial transcript, and
commit if the final transcript still matches, discard if it changed. Done blindly, it either wastes
LLM calls or leaves latency on the table.

**Jaldi learns the policy: at each moment of a user's turn, is this partial safe to start
answering from?**

Headline artifact: a **Pareto curve — latency saved (p50/p90 ms) vs wasted LLM calls per turn**,
learned policy vs standard heuristics.

**The India twist (core hypothesis, must be measured, not assumed):** Hindi and Kannada are
verb-final (SOV). Meaning-changing words often arrive last:
"Mujhe kal ki booking cancel… **nahi** karni". English-tuned speculation heuristics may fail more
often on Hindi/Hinglish. Measure the "late meaning-flip rate" per language and show whether a
learned policy closes the gap.

Audience: ML engineers at voice-AI companies. Rigorous, honest, reproducible.

## Hard rules

1. **No invented numbers.** Every number in README, charts, thread and demo is generated from
   `results/*.json` by `python -m jaldi.cli report`. If something fails or the hypothesis is not
   supported, say so plainly.
2. **No leakage.** Splits are speaker-disjoint and scenario-disjoint. Features may only use
   information available *at that moment of the turn in production* — never the final transcript,
   total turn length, or synthesis metadata.
3. **Sanity gates** end every stage. If one fails, stop, log it in PROGRESS.md, propose a fix.
4. **Time-box** each stage; take the fallback if it overruns by ~50%.
5. Commit after each stage: `stage N: <summary>`. Fixed seeds. Log configs.
6. Keep `PROGRESS.md` current: done / key numbers / issues / next step.

## Environment

- **Local laptop (no CUDA):** code, CPU stages, analysis, demo. Python 3.11, `uv`,
  `pyproject.toml`, `ruff`, `pytest`, `ffmpeg`.
- **GPU = Google Colab A100:** TTS, ASR partials, local LLM labelling. Create
  `notebooks/gpu_stages.ipynb` that clones the repo, installs deps, runs CLI GPU stages and writes
  to Drive `jaldi_data/`. The human runs it and syncs outputs into `data/`.
- `.env`: `ANTHROPIC_API_KEY` (corpus generation). Optional streaming-STT key
  (`DEEPGRAM_API_KEY` / `ASSEMBLYAI_API_KEY` / `SONIOX_API_KEY`) — used only to validate the partial
  simulation in Stage 3; skip if absent.

## Repo layout

```
jaldi/
  jaldi/
    cli.py                  # corpus, tts, partials, label, features, train, simulate, report, all
    corpus/   generate.py validate.py
    audio/    tts.py assemble.py prosody.py
    asr/      partials.py streaming_api.py
    labels/   transcript_match.py action_match.py
    features/ text.py stability.py acoustic.py turn_detector.py
    models/   gbm.py text_encoder.py fusion.py calibrate.py
    sim/      policy.py simulate.py baselines.py
    eval/     metrics.py plots.py report.py
  app/demo.py               # Gradio
  notebooks/gpu_stages.ipynb
  data/                     # gitignored
  results/                  # json + png, committed
  docs/                     # thread.md, demo_script.md, integration.md
  tests/
  PROGRESS.md
  README.md
```

---

## Stage 1 — User-turn corpus (target 2h, CPU)

Generic voice-agent scenarios (no single industry): restaurant/clinic booking, order status,
rescheduling, FAQ/support, account changes, travel queries. 6–8 scenarios, each with a small
**action schema** (e.g. `{intent, slots}`) used later for labels.

**Languages:** English (~35%), Hindi (~35%), Hinglish code-mixed (~30%).

Claude generates user turns as **phrase segments** (so pauses can be inserted between them), with
controlled phenomena (~40% of turns):
- Late negation / polarity flip ("…cancel nahi karni", "…I don't want that")
- Late correction ("Tuesday… no, Wednesday", "do baje… nahi, teen baje")
- Trailing qualifier that changes the action ("book it… but only if it's a window seat")
- Mid-turn hesitation that looks complete ("I want to book a table. [pause] For six people.")
- Plain complete turns (the rest)

Output `data/corpus.jsonl`:
`id, scenario, lang, segments:[text], final_text, phenomenon, gold_action{intent,slots}, split`.

**Gate:** ≥1,000 turns; every turn has a valid gold action; phenomenon counts logged per language.

---

## Stage 2 — Audio with realistic pauses (target 2h, GPU)

- TTS: `ai4bharat/indic-parler-tts` (supports Indic languages and English). Use named speakers from
  the model card; vary pace/pitch via caption. Hold out ≥1 speaker per language for test.
- Synthesize **each segment separately**, then join with sampled silences: short (150–400 ms) and
  long hesitations (500–1200 ms). Keep exact segment timestamps.
- Script check first: for Hinglish, try native-script-with-Latin-English vs romanized on 10 turns,
  pick the format with better ASR round-trip, log the decision.
- Optional: light telephony degradation (8 kHz, μ-law, mild noise) on 50% of clips.
- Benchmark throughput on 20 turns, scale N to fit ~90 GPU-minutes.

**Gate:** clean-audio final-transcript WER per language reported; drop unintelligible clips, log count.

---

## Stage 3 — Simulated streaming partials (target 2h, GPU)

- Emit a partial every 200 ms of audio: run ASR on the audio prefix up to t.
  - English + Hinglish: `openai/whisper-large-v3-turbo`
  - Hindi: also `ai4bharat/indic-conformer-600m-multilingual` (`trust_remote_code=True`);
    pick the better one per language for the main run.
- Store per partial: `t_ms`, text, ASR confidence if available, silence-so-far (from energy VAD).
- End-of-turn reference time = end of last speech segment + a fixed endpointing delay
  (configurable, default 500 ms — typical VAD-silence setting).
- **Fidelity check (if a streaming-STT key exists):** run ~100 turns through a real streaming API
  with interim results and compare partial revision statistics vs the simulation. Report it as a
  limitation either way.

**Gate:** partial count per turn sane; final partial ≈ full-clip transcript.

---

## Stage 4 — Labels (target 2h)

Candidate speculation points = every partial within a turn.

- **L1 — transcript-safe:** normalized(partial) == normalized(final transcript). Speculation from
  here is reusable verbatim. Cheap, exact.
- **L2 — action-safe:** the next action an LLM derives from the partial equals the action derived
  from the final transcript (`{intent, slots}` exact match). Use a local open instruct model via
  vLLM on the A100 (e.g. Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, or newer; must handle
  Hindi), temperature 0, JSON output against the scenario schema.
- Sanity: action from *final transcript* must match `gold_action` on ≥90% of turns; otherwise fix
  prompts before trusting L2.

**Key analysis (milestone M1 — priority):**
- **Late meaning-flip rate** per language: share of turns where the L2 label flips from safe to
  unsafe in the last 1–2 words / last segment.
- Heuristic baseline (speculate at first pause ≥ X ms) evaluated per language.
- Plot 1: meaning-flip rate and heuristic waste by language (English vs Hindi vs Hinglish).

> **M1 is the first thing to show. Reach it before building models.**

---

## Stage 5 — Features (target 1.5h, CPU)

All features use only information available at time t:

| Group | Features |
|---|---|
| Text | words so far; ends with conjunction / filler / postposition / auxiliary verb; ends with a verb (Hindi verb-final cue); open quote/number; question word present |
| ASR stability | edit distance between last 2 partials; # revisions in last k partials; ms since last change; ASR confidence |
| Acoustic/prosodic | silence so far; F0 slope and energy drop over last 300 ms (finality); speaking rate |
| Turn detector | probability from an open turn-detection model (pipecat smart-turn latest open release; optionally the open Smart Turn Hinglish model) |
| Meta | language / code-mix index, scenario |

**Gate:** unit tests prove no feature reads the future (test on truncated inputs).

---

## Stage 6 — Models (target 2.5h)

- **A.** LightGBM on Stage-5 features
- **B.** Small multilingual text encoder fine-tuned on partial text → P(safe)
  (e.g. `google/muril-base-cased` or `xlm-roberta-base`; GPU if needed)
- **C.** Fusion: B's score as a feature into A
- Isotonic calibration on val. Grouped CV or 5 seeds for CIs. Train separate heads for L1 and L2;
  report both.
- Metrics: AUROC, ECE, Brier, per language.

**Gate:** learned model beats the best single heuristic on AUROC with non-overlapping CIs —
otherwise report honestly and analyze why.

---

## Stage 7 — Policy simulation (target 2h, CPU)

Replay each test turn in time:
- Policy speculates at the first partial where calibrated P(safe) ≥ τ. If a later partial changes
  the transcript (L1) or action (L2), discard and allow re-speculation (max k, default 3).
- At end-of-turn: commit if the speculated partial is still valid, else start fresh.
- LLM "time-to-ready" for the speculated response = measured TTFT + generation time from the local
  LLM in Stage 4 (store measured distribution; also run with a configurable constant).

Metrics per policy:
- **Latency saved** = min(time-to-ready, end_of_turn − speculation_start) for committed
  speculations; report p50/p90 voice-to-voice improvement
- **Waste** = discarded speculative calls per turn, and wasted tokens
- **Wrong-commit rate** (L2 mode only — committing an action that differs from the final one).
  Must be reported; target ≈ 0.

Baselines: never speculate; speculate on every partial; first pause ≥ X ms (sweep X);
turn-detector ≥ τ only; "stable for k partials" (sweep k).

**Plot 2 (headline):** Pareto front — latency saved vs waste, all policies, per language.
**Plot 3:** a cost-weighted view: `cost = c_call × wasted_calls − c_ms × ms_saved` for a few
cost settings; show the best τ per language.

---

## Stage 8 — Demo (target 2h, CPU)

Gradio app (`app/demo.py`), must run on the laptop:
- **Replay mode (default):** pick a test turn → animated timeline: audio waveform, partials as they
  arrive, Jaldi P(safe) line, heuristic trigger vs Jaldi trigger, end-of-turn marker,
  "ms saved / call wasted" badges. Precomputed results, no GPU needed.
- **Live mode:** mic → chunked partials with a small Whisper on CPU (label it "CPU demo model") or
  a streaming API if a key exists → live P(safe) line.
- Include 3 curated examples per language, including at least one late-negation case where the
  heuristic speculates wrongly and Jaldi waits.

---

## Stage 9 — Write-up (target 1.5h)

`README.md` (numbers injected by `jaldi.cli report`):
1. Problem in one paragraph + latency waterfall sketch
2. The verb-final hypothesis + Plot 1
3. Method diagram: partials → features → calibrated P(safe) → speculate/commit/discard
4. Results: Plot 2, Plot 3, ablation by feature group, L1 vs L2
5. Why labels are free in production: every call log already contains interim and final
   transcripts, so this trains on a company's own logs with no annotation
6. Beyond cascades: speech-to-speech models face the same question (when to commit to speaking);
   note as discussion, not a claim
7. Limitations: synthetic TTS voices and inserted pauses, simulated partials (with fidelity check
   result), single LLM for labels, generic scenarios
8. Reproduce: exact commands

Also: `docs/thread.md` (6 tweets), `docs/demo_script.md` (90 s).

---

## Stretch (only after Stage 9)

- `docs/integration.md`: where Jaldi plugs into open-source stacks — read `bolna-ai/bolna` and
  pipecat source, find the hook between interim transcripts and LLM call, write a minimal example.
  No PR unless clean.
- Real mini test set: ~50 turns recorded by humans (English/Hindi/Hinglish) with natural pauses.
- Kannada.

## Milestones

| Milestone | Contents | Cumulative target |
|---|---|---|
| M1 | Corpus + audio + partials + labels + Plot 1 (meaning-flip by language) | ~8h |
| M2 | Features + models + calibration | ~12h |
| M3 | Policy simulation + Pareto (Plot 2/3) | ~14h |
| M4 | Demo + README + thread | ~17.5h |
| Buffer | GPU sync, fixes, stretch | ~6h |
