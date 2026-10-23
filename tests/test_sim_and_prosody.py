import numpy as np

from jaldi.asr.text_sim import simulate_turn
from jaldi.audio.prosody import end_prosody, silence_and_speech_ms
from jaldi.config import Config
from jaldi.corpus.generate import generate
from jaldi.eval.metrics import auroc, ece
from jaldi.sim.simulate import TurnView, run_turn


def _view(keys, t, final_key, eot):
    n = len(keys)
    return TurnView(id="x", lang="en", t=np.array(t, float), keys_l1=keys, keys_l2=keys,
                    silence=np.zeros(n), stable=np.zeros(n), turn_p=np.full(n, np.nan), eot=eot,
                    speech_end=eot - 500, final_key_l1=final_key, final_key_l2=final_key,
                    final_matches_gold=True)


def test_commit_saves_min_of_ttr_and_time_left():
    v = _view(["a", "ab", "abc", "abc"], [200, 400, 600, 1200], "abc", 1200)
    r = run_turn(v, np.array([0, 0, 1, 1], bool), "l1", ttr=500, max_spec=3)
    assert r["committed"] and r["saved"] == 500 and r["wasted"] == 0
    r = run_turn(v, np.array([0, 0, 0, 1], bool), "l1", ttr=500, max_spec=3)
    assert r["saved"] == 0  # speculating exactly at end of turn saves nothing


def test_invalidated_speculation_is_waste():
    v = _view(["a", "ab", "abc"], [200, 400, 900], "abc", 900)
    r = run_turn(v, np.array([1, 1, 0], bool), "l1", ttr=600, max_spec=3)
    assert r["wasted"] == 2 and not r["committed"]
    r = run_turn(v, np.array([1, 1, 0], bool), "l1", ttr=600, max_spec=1)
    assert r["wasted"] == 1 and r["calls"] == 1


def test_text_sim_final_partial_is_final_transcript():
    cfg = Config(n_turns=30)
    for t in generate(cfg):
        rec = simulate_turn(t, cfg)
        assert rec["partials"][-1]["text"] == rec["final_text"] == t["final_text"]
        assert rec["partials"][-1]["t_ms"] == rec["eot_ms"]
        ts = [p["t_ms"] for p in rec["partials"]]
        assert ts == sorted(ts)


def test_prosody_on_synthetic_audio():
    sr = 16000
    t = np.arange(int(0.8 * sr)) / sr
    f0 = 220 * np.exp(-0.8 * t)  # falling pitch
    speech = 0.5 * np.sin(2 * np.pi * np.cumsum(f0) / sr) * np.linspace(1, 0.4, len(t))
    x = np.concatenate([np.zeros(int(0.2 * sr)), speech, np.zeros(int(0.4 * sr))]).astype(np.float32)
    sil, sp = silence_and_speech_ms(x, sr)
    assert 300 <= sil <= 460 and sp >= 1000
    slope, drop = end_prosody(x, sr)
    assert slope < 0 and drop > 0


def test_metrics():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ece([0, 1], [0.0, 1.0]) == 0.0
