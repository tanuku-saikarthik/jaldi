# Where Jaldi plugs into an open-source voice stack (Bolna)

Read from [`bolna-ai/bolna`](https://github.com/bolna-ai/bolna) at commit `7de6108` (2026-10-23, MIT).
Line numbers refer to that commit. I did not read pipecat for this note.

## What Bolna already does

Bolna already speculates. The path is:

1. **`bolna/transcriber/deepgram_transcriber.py` ~L998–1035.** Deepgram Flux sends `EagerEndOfTurn`
   with a transcript and a confidence. The transcriber turns it into
   `{"type": "eager_end_of_turn", "content": transcript, "confidence": ...}`. Flux only sends it when
   `eager_eot_threshold` is configured (L131–133).
2. **`bolna/agent_manager/task_manager.py` ~L5885–5939.** On `eager_end_of_turn`:
   - it skips if the confidence is below `eager_eot_threshold`, or if audio is playing / a response is in flight;
   - otherwise it starts `self.eager_llm_task = asyncio.create_task(self._run_llm_task(...))` on the
     eager transcript and appends a provisional user row to history.
3. **Same file ~L5941–5953.** On `turn_resumed` (the caller kept talking), it cancels `eager_llm_task`
   and reverts the speculative history rows. This is the *waste* case in Jaldi's terms.
4. **Same file ~L6009+.** On the final `EndOfTurn` with `was_eager`, the eager reply is used. This is the *commit* case.
5. **`bolna/transcriber/realtime_transcriber.py` ~L235.** The OpenAI Realtime transcriber also maps
   `input_audio_buffer.eager_end_of_turn` to the same event, with `confidence: None`.

So the decision "speculate now?" is a single threshold on the recognizer's end-of-turn
confidence. It only exists for recognizers that emit an eager event.

## Two ways to add Jaldi

**A. Gate the existing eager path.** In the `eager_end_of_turn` branch, after the confidence check, add:

```python
# task_manager.py, inside the eager_end_of_turn branch, before starting eager_llm_task
p_safe = self.jaldi.score_partial(
    text=eager_transcript,
    lang=self.language,                       # or the LID result
    silence_ms=time_since_last_speech_ms,      # from the VAD / interim timestamps
    history=self.recent_interims,              # interims seen this turn (for stability features)
    eot_confidence=eot_confidence,             # becomes one more feature
)
if p_safe < self.jaldi_tau:
    logger.info(f"Skipping speculative LLM: Jaldi P(safe)={p_safe:.2f} < {self.jaldi_tau}")
else:
    ...  # existing code that creates eager_llm_task
```

This keeps Flux's signal and adds the language-aware part. One example is "the verb just
arrived, a Hindi negation may follow".

**B. Create eager events for any recognizer.** For recognizers without an eager event, run
Jaldi on every `interim_transcript_received` (~L5763). When P(safe) crosses τ, emit the same
internal `eager_end_of_turn`. If a later interim changes the transcript or action, emit `turn_resumed`.
The rest of the speculation machinery (cancel, history revert, commit on `was_eager`) is reused unchanged.

## What it needs at runtime

- **Inputs:** the interim text, a timestamp per interim, and silence-so-far from the VAD.
  Optionally a turn-detector score and prosody.
- **Model:** a LightGBM ensemble plus an isotonic table (`jaldi.models.scorer.JaldiScorer`). It costs a
  few milliseconds on CPU per interim, with no LLM call.
- **Training data:** call logs that already store interims and the final transcript. L1 labels need
  nothing else. L2 labels need one offline extraction pass per unique interim.

None of this is a PR. It is a map of the hook, for discussion.
