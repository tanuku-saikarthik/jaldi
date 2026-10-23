# 90-second demo script

> **DRAFT — generated from the smoke run** (template corpus, simulated partials). Do not post these numbers as real results. Regenerate with `jaldi report` after the GPU run.

**0–15 s. The problem.** "Voice agents start thinking only after you stop. Speculative generation starts early, but firing at the wrong moment wastes LLM calls."

**15–40 s. Replay a Hindi/Hinglish negation turn** (demo → Replay).
Example: "अपॉइंटमेंट कैंसल मत कीजिए". Point at where the pause heuristic fires (wasted 2 call(s), saved 276 ms) vs Jaldi (wasted 0, saved 476 ms).

**40–60 s. Plot 1.** "Meaning flips late in Hindi because negation comes after the verb, but the action also settles earlier. A pause rule can't tell these apart."

**60–80 s. Plot 2.** "Tuned for ≤1 wasted call per turn: Jaldi saves 482 ms per turn (0.68 wasted) vs the best pause rule 191 ms (0.97 wasted), on scenarios it never saw."

**80–90 s. Close.** "Labels come free from call logs. It's a few ms on CPU. It would slot in where Bolna already starts its eager LLM task (docs/integration.md)."
