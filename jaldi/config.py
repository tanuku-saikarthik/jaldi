"""Single source of truth for pipeline settings. Every run logs the config it used."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    seed: int = 13
    data_dir: str = "data"
    results_dir: str = "results"

    # ---- Stage 1: corpus
    n_turns: int = 1800
    lang_mix: dict = field(default_factory=lambda: {"en": 0.35, "hi": 0.35, "hinglish": 0.30})
    phenomenon_mix: dict = field(
        default_factory=lambda: {
            "plain": 0.60,
            "negation": 0.10,
            "late_correction": 0.10,
            "trailing_qualifier": 0.10,
            "hesitation": 0.10,
        }
    )
    # scenario-disjoint split (hard rule 2)
    val_scenarios: list = field(default_factory=lambda: ["delivery"])
    test_scenarios: list = field(default_factory=lambda: ["clinic", "cab"])

    # ---- pauses between segments (same distribution for every language)
    none_pause_ms: tuple = (20, 100)
    short_pause_ms: tuple = (150, 400)
    long_pause_ms: tuple = (500, 1200)
    # pause type at ordinary chunk boundaries; hesitation boundaries are always long
    pause_probs: dict = field(default_factory=lambda: {"none": 0.45, "short": 0.35, "long": 0.20})

    # ---- Stage 3: partials
    partial_step_ms: int = 200
    endpoint_delay_ms: int = 500  # end-of-turn = end of last speech + this (typical VAD silence setting)

    # text-simulation mode (CPU smoke run only; not a substitute for real ASR)
    sim_asr_lag_ms: int = 120  # a word appears this long after it ends
    sim_p_early_word: float = 0.3  # in-progress word shown early (as a prefix)
    sim_p_revision: float = 0.12  # last word temporarily wrong, fixed at the next partial

    # ---- Stage 7: policy simulation
    time_to_ready_ms: float = 600.0  # ASSUMPTION: LLM TTFT + first TTS chunk; replace with measured
    time_to_ready_sensitivity: list = field(default_factory=lambda: [300.0, 600.0, 1000.0])
    max_speculations: int = 3
    pause_sweep_ms: list = field(default_factory=lambda: [100, 200, 300, 400, 500, 600, 800])
    stable_sweep_k: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    cost_per_wasted_call_ms: list = field(default_factory=lambda: [100.0, 300.0, 1000.0])
    waste_budgets: list = field(default_factory=lambda: [0.25, 0.5, 1.0])

    # ---- Stage 6
    bootstrap: int = 1000

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        cfg = cls()
        if path:
            for k, v in json.loads(Path(path).read_text()).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    @property
    def data(self) -> Path:
        return Path(self.data_dir)

    @property
    def results(self) -> Path:
        return Path(self.results_dir)
