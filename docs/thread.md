# X thread

> **DRAFT — generated from the smoke run** (template corpus, simulated partials). Do not post these numbers as real results. Regenerate with `jaldi report` after the GPU run.

1/ Voice agents wait for you to stop talking, then start thinking. Speculative generation starts the LLM on the partial transcript instead. The hard part is *when*. I built Jaldi: a tiny calibrated model that decides, at every streaming partial, whether it's safe to start. 🧵

2/ Why Indian languages are interesting: Hindi is verb-final. "booking cancel… nahi karni" flips meaning on the last word. English says "I don't want to cancel" up front.

3/ Measured on a synthetic template corpus with simulated ASR partials (smoke run): late meaning flips in 8.4% of English turns vs 14.0% Hindi and 17.2% Hinglish. But the action also settles earlier in Hindi: median headroom 827 ms vs 350 ms in English. More risk AND more room.

4/ A pause rule can't see word order. Tuned for ≤1 wasted LLM call per turn: Jaldi saves 313 / 574 / 577 ms per turn (EN / HI / Hinglish) wasting 0.68 calls; the best pause threshold saves 196 / 189 / 187 ms wasting 0.97. Test scenarios never seen in training.

5/ The nice part: labels are free. Every call log already has interim and final transcripts, so you can train this on your own traffic with zero annotation. It runs in milliseconds on CPU, no LLM call.

6/ Caveats are in the README (synthetic voices, simulated streaming, one labeller). Code, plots, Colab notebook for the GPU run: github.com/tanuku-saikarthik/jaldi. Would love feedback from people running voice agents in production.
