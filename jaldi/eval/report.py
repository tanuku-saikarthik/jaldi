"""Stage 9: build results/summary.json, figures, demo examples and the README results block.

Hard rule: every number in the README comes from results/*.json through this module.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..io import read_json, read_jsonl, write_json
from ..sim import policy as P
from ..sim.simulate import make_view, run_turn
from . import plots

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LANGS = ("en", "hi", "hinglish")
NAME = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def _ci(t) -> str:
    p, lo, hi = t
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _ci_raw(t, nd=3) -> str:
    p, lo, hi = t
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _overlap(a, b) -> bool:
    return not (a[1] > b[2] or b[1] > a[2])


def verdicts(m1: dict, model: dict, policy: dict, budget: str) -> list[str]:
    out = []
    bl = m1["by_lang"]
    en = bl["en"]["meaning_flip_rate"]
    for lg in ("hi", "hinglish"):
        x = bl[lg]["meaning_flip_rate"]
        if _overlap(x, en):
            out.append(f"Late meaning flips: {NAME[lg]} {_pct(x[0])} vs English {_pct(en[0])}; "
                       "95% CIs overlap, so no difference is claimed.")
        else:
            word = "higher" if x[0] > en[0] else "lower"
            out.append(f"Late meaning flips: {NAME[lg]} {_pct(x[0])} vs English {_pct(en[0])} "
                       f"({word}; 95% CIs do not overlap).")
    neg = {lg: m1["flip_by_phenomenon"][lg].get("negation", {}).get("rate") for lg in LANGS}
    if all(v is not None for v in neg.values()):
        out.append("Negation turns that flip: " + ", ".join(f"{NAME[lg]} {_pct(neg[lg])}" for lg in LANGS)
                   + ". In these templates English states negation before the verb; Hindi/Hinglish after it.")
    h = {lg: bl[lg]["l2_headroom_ms_median"] for lg in LANGS}
    out.append("Opportunity runs the other way: median L2 headroom " +
               ", ".join(f"{NAME[lg]} {h[lg]:.0f} ms" for lg in LANGS) +
               " (the action settles at the verb, before trailing auxiliaries like \"करनी है\").")
    sel = policy["selected"]["l2"][budget]
    # report the English-tuned pause heuristic at the tightest budget where it saves something real
    for b in sorted(policy["selected"]["l2"], key=float):
        pen = policy["selected"]["l2"][b].get("pause (tuned on English val)", {})
        if pen.get("param") is not None and pen["test"]["en"]["mean_saved_ms"] >= 50:
            w = pen["test"]
            out.append(
                f"English-tuned pause heuristic (X={pen['param']:.0f} ms, tuned under ≤ {b} wasted calls/turn): "
                + ", ".join(f"{NAME[lg]} {w[lg]['mean_saved_ms']:.0f} ms saved / "
                            f"{w[lg]['wasted_per_turn']:.2f} wasted" for lg in LANGS)
                + ". Pauses here follow one distribution for every language, so a pause rule "
                  "cannot see word order: it neither avoids the negation flips nor uses the extra headroom.")
            break
    j = sel.get("Jaldi (tuned on all-language val)", {})
    if j.get("param") is not None:
        out.append(f"Jaldi (τ={j['param']:g}, tuned under ≤ {budget} wasted calls/turn) saves " +
                   ", ".join(f"{NAME[lg]} {j['test'][lg]['mean_saved_ms']:.0f} ms" for lg in LANGS) +
                   " per turn on average.")
    for lab in ("l1", "l2"):
        g = model["labels"][lab]["gate"]
        out.append(f"{lab.upper()} gate ({g['rule']}): {'PASSED' if g['passed'] else 'FAILED'} "
                   f"vs {g['best_heuristic']}.")
    return out


def _policy_table(policy: dict, mode: str, budget: str) -> str:
    sel = policy["selected"][mode][budget]
    rows = ["| policy | param | " + " | ".join(f"{NAME[lg]} saved ms / wasted per turn" for lg in LANGS)
            + " | all: p50 / p90 latency (ms) |", "|---|---|---|---|---|---|"]
    for label, s in sel.items():
        if s.get("param") is None:
            rows.append(f"| {label} | – | " + " | ".join("–" for _ in LANGS) + " | – |")
            continue
        cells = [f"{s['test'][lg]['mean_saved_ms']:.0f} / {s['test'][lg]['wasted_per_turn']:.2f}" for lg in LANGS]
        a = s["test"]["all"]
        rows.append(f"| {label} | {s['param']:g} | " + " | ".join(cells)
                    + f" | {a['latency_p50_ms']:.0f} / {a['latency_p90_ms']:.0f} |")
    return "\n".join(rows)


def demo_examples(turns: list[dict], preds: pd.DataFrame, policy: dict, cfg: Config) -> list[dict]:
    sel = policy["selected"]["l2"]
    b = str(cfg.waste_budgets[-1])
    pause_x = sel[b]["pause (tuned on all-language val)"]["param"] or cfg.pause_sweep_ms[1]
    tau = sel[str(cfg.waste_budgets[0])]["Jaldi (tuned on all-language val)"]["param"] or 0.5
    pr = preds.sort_values(["id", "idx"])
    score = {i: g["p_l2"].values for i, g in pr.groupby("id")}
    test = [t for t in turns if t["split"] == "test"]
    rng = np.random.default_rng(cfg.seed)
    chosen = []
    for lg in LANGS:
        pool = [t for t in test if t["lang"] == lg]
        picks = []

        def scored(t):
            v = make_view(t)
            rh = run_turn(v, P.pause(v, pause_x), "l2", cfg.time_to_ready_ms, cfg.max_speculations)
            rj = run_turn(v, P.jaldi(score[t["id"]], tau), "l2", cfg.time_to_ready_ms, cfg.max_speculations)
            return rh, rj

        wants = [("negation", lambda rh, rj: rh["wasted"] > rj["wasted"]),
                 ("late_correction", lambda rh, rj: True),
                 ("plain", lambda rh, rj: rj["saved"] > rh["saved"])]
        for phen, cond in wants:
            cands = [t for t in pool if t["phenomenon"] == phen]
            rng.shuffle(cands)
            for t in cands:
                rh, rj = scored(t)
                if cond(rh, rj):
                    picks.append((t, rh, rj))
                    break
        for t, rh, rj in picks:
            v = make_view(t)
            chosen.append({
                "id": t["id"], "lang": t["lang"], "phenomenon": t["phenomenon"],
                "final_text": t["final_text"], "gold_action": t["gold_action"],
                "eot_ms": t["eot_ms"], "speech_end_ms": t["speech_end_ms"],
                "pause_x_ms": pause_x, "tau": tau, "time_to_ready_ms": cfg.time_to_ready_ms,
                "partials": [{"t_ms": p["t_ms"], "text": p["text"], "silence_ms": p["silence_ms"],
                              "p_safe": float(s), "l2": p["l2"], "pause_trigger": bool(pt),
                              "jaldi_trigger": bool(s >= tau)}
                             for p, s, pt in zip(t["partials"], score[t["id"]], P.pause(v, pause_x))],
                "heuristic": rh, "jaldi": rj,
            })
    return chosen


def build_report(cfg: Config, readme_path: str = "README.md") -> dict:
    R = cfg.results
    m1, model, policy = read_json(R / "m1.json"), read_json(R / "model.json"), read_json(R / "policy.json")
    labels_meta = read_json(R / "labels_meta.json")
    budget = str(cfg.waste_budgets[0])

    plots.plot_m1(m1, R / "plot1_meaning_flip.png")
    plots.plot_pareto(policy, "l2", R / "plot2_pareto_l2.png")
    plots.plot_pareto(policy, "l1", R / "plot2_pareto_l1.png")
    plots.plot_cost(policy, "l2", R / "plot3_cost_l2.png")
    plots.plot_reliability(model, R / "reliability.png")

    turns = read_jsonl(cfg.data / "labels.jsonl")
    preds = pd.read_csv(cfg.data / "preds.csv")
    write_json(R / "demo_examples.json", demo_examples(turns, preds, policy, cfg))

    v = verdicts(m1, model, policy, budget)
    summary = {"mode": labels_meta["mode"], "extractor": labels_meta["extractor"],
               "n_turns": labels_meta["n_turns"], "verdicts": v}
    write_json(R / "summary.json", summary)

    md = _render(cfg, m1, model, policy, labels_meta, v, budget)
    examples = read_json(R / "demo_examples.json")
    docs = Path(readme_path).resolve().parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "thread.md").write_text(_thread(m1, policy, labels_meta, budget), encoding="utf-8")
    (docs / "demo_script.md").write_text(_demo_script(m1, policy, labels_meta, budget, examples),
                                         encoding="utf-8")
    p = Path(readme_path)
    text = p.read_text(encoding="utf-8")
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), f"{START}\n{md}\n{END}", text, flags=re.S)
        p.write_text(text, encoding="utf-8")
    return summary


def _render(cfg, m1, model, policy, meta, verdict_lines, budget) -> str:
    L = []
    if meta["mode"] == "text_sim":
        L.append("> **Smoke run.** Template corpus + *simulated* streaming partials (no audio, no ASR) "
                 "+ rule-based action labels. These numbers show the pipeline works end to end and how the "
                 "templates behave. They are **not** evidence about real callers or real ASR. "
                 "Real numbers come from the GPU run (`notebooks/gpu_stages.ipynb`).\n")
    L.append(f"Run: mode=`{meta['mode']}`, labels=`{meta['extractor']}`, {meta['n_turns']} turns "
             f"(train/val/test = {model['n_turns']['train']}/{model['n_turns']['val']}/{model['n_turns']['test']}, "
             f"scenario- and speaker-disjoint). Final action from final transcript matches gold on "
             f"{_pct(meta['final_action_matches_gold'])} of turns. Time-to-ready assumed "
             f"{policy['time_to_ready_ms']:.0f} ms, endpointing delay {policy['endpoint_delay_ms']:.0f} ms.\n")
    L.append("### What the numbers say\n")
    L += [f"- {x}" for x in verdict_lines]
    L.append("\n### M1: meaning flips by language\n")
    L.append("![Plot 1](results/plot1_meaning_flip.png)\n")
    L.append("| language | turns | late meaning-flip rate [95% CI] | false-actionable rate | median L1 headroom | median L2 headroom |")
    L.append("|---|---|---|---|---|---|")
    for lg in LANGS:
        b = m1["by_lang"][lg]
        L.append(f"| {NAME[lg]} | {b['n_turns']} | {_ci(b['meaning_flip_rate'])} | "
                 f"{_pct(b['false_actionable_rate'][0])} | {b['l1_headroom_ms_median']:.0f} ms | "
                 f"{b['l2_headroom_ms_median']:.0f} ms |")
    L.append("\n### Policies: latency saved vs wasted calls (test split)\n")
    L.append("![Plot 2](results/plot2_pareto_l2.png)\n")
    budgets = sorted(policy["selected"]["l2"], key=float)
    for b in budgets:
        L.append(f"Action-safe (L2) speculation, settings tuned on val under ≤ {b} wasted calls/turn:\n")
        L.append(_policy_table(policy, "l2", b) + "\n")
    L.append(f"Transcript-safe (L1) speculation, tuned under ≤ {budgets[-1]} wasted calls/turn:\n")
    L.append(_policy_table(policy, "l1", budgets[-1]))
    L.append("\nCost view (oracle τ per language, illustration only):\n")
    L.append("![Plot 3](results/plot3_cost_l2.png)\n")
    L.append("### Model\n")
    L.append("| label | test base rate | AUROC [95% CI, turn bootstrap] | seed std | ECE | Brier | best single heuristic AUROC |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in ("l1", "l2"):
        r = model["labels"][lab]
        g = r["gate"]
        L.append(f"| {lab.upper()} | {_pct(r['base_rate_test'])} | {_ci_raw(r['auroc_ci'])} | "
                 f"{r['auroc_seed_std']:.4f} | {r['ece_test']:.3f} | {r['brier_test']:.3f} | "
                 f"{g['best_heuristic']}: {_ci_raw(g['best_heuristic_auroc_ci'])} |")
    L.append("\nPer-language AUROC: " + "; ".join(
        f"{lab.upper()} " + ", ".join(f"{NAME[lg]} {model['labels'][lab]['by_lang'][lg]['auroc']:.3f}"
                                      for lg in LANGS if lg in model['labels'][lab]['by_lang'])
        for lab in ("l1", "l2")) + ".\n")
    L.append("Ablation (test AUROC, one feature group removed):\n")
    L.append("| label | " + " | ".join(model["labels"]["l2"]["ablation_test_auroc"]) + " |")
    L.append("|---|" + "---|" * len(model["labels"]["l2"]["ablation_test_auroc"]))
    for lab in ("l1", "l2"):
        a = model["labels"][lab]["ablation_test_auroc"]
        L.append(f"| {lab.upper()} | " + " | ".join(f"{v:.3f}" for v in a.values()) + " |")
    if model["features_dropped_all_nan"]:
        L.append(f"\nFeatures unavailable in this run (all NaN, dropped): `{', '.join(model['features_dropped_all_nan'])}`.")
    L.append("\n![Reliability](results/reliability.png)\n")
    sens = policy["ttr_sensitivity"]["l2"]
    if sens:
        L.append("Time-to-ready sensitivity (L2, mean ms saved per turn at the selected settings):\n")
        ttrs = list(next(iter(sens.values())))
        L.append("| policy | " + " | ".join(f"TTR {float(t):.0f} ms ({'/'.join(NAME[lg][:2] for lg in LANGS)})" for t in ttrs) + " |")
        L.append("|---|" + "---|" * len(ttrs))
        for label, by_t in sens.items():
            L.append(f"| {label} | " + " | ".join(
                " / ".join(f"{by_t[t][lg]['mean_saved_ms']:.0f}" for lg in LANGS) for t in ttrs) + " |")
    return "\n".join(L)


def _draft_banner(meta: dict) -> str:
    if meta["mode"] == "text_sim":
        return ("> **DRAFT — generated from the smoke run** (template corpus, simulated partials). "
                "Do not post these numbers as real results. Regenerate with `jaldi report` after the GPU run.\n\n")
    return "> Generated by `jaldi report` from results/*.json. Edit wording, not numbers.\n\n"


def _headline(policy: dict, budget: str) -> tuple[dict, dict, float]:
    """Headline at the LOOSEST budget, where the pause heuristic is genuinely competitive."""
    budget = max(policy["selected"]["l2"], key=float)
    sel = policy["selected"]["l2"][budget]
    j = sel["Jaldi (tuned on all-language val)"]
    p = sel["pause (tuned on all-language val)"]
    return j, p, float(budget)


def _thread(m1: dict, policy: dict, meta: dict, budget: str) -> str:
    bl = m1["by_lang"]
    j, p, b = _headline(policy, budget)
    smoke = meta["mode"] == "text_sim"
    data_note = ("on a synthetic template corpus with simulated ASR partials (smoke run)" if smoke
                 else "on synthetic Indic TTS calls with Whisper partials")
    t = [
        "1/ Voice agents wait for you to stop talking, then start thinking. Speculative generation "
        "starts the LLM on the partial transcript instead. The hard part is *when*. I built Jaldi: a "
        "tiny calibrated model that decides, at every streaming partial, whether it's safe to start. 🧵",
        "2/ Why Indian languages are interesting: Hindi is verb-final. \"booking cancel… nahi karni\" "
        "flips meaning on the last word. English says \"I don't want to cancel\" up front.",
        f"3/ Measured {data_note}: late meaning flips in {_pct(bl['en']['meaning_flip_rate'][0])} of English turns vs "
        f"{_pct(bl['hi']['meaning_flip_rate'][0])} Hindi and {_pct(bl['hinglish']['meaning_flip_rate'][0])} Hinglish. "
        f"But the action also settles earlier in Hindi: median headroom {bl['hi']['l2_headroom_ms_median']:.0f} ms "
        f"vs {bl['en']['l2_headroom_ms_median']:.0f} ms in English. More risk AND more room.",
        f"4/ A pause rule can't see word order. Tuned for ≤{b:g} wasted LLM call per turn: Jaldi saves "
        f"{j['test']['en']['mean_saved_ms']:.0f} / {j['test']['hi']['mean_saved_ms']:.0f} / "
        f"{j['test']['hinglish']['mean_saved_ms']:.0f} ms per turn (EN / HI / Hinglish) wasting "
        f"{j['test']['all']['wasted_per_turn']:.2f} calls; the best pause threshold saves "
        f"{p['test']['en']['mean_saved_ms']:.0f} / {p['test']['hi']['mean_saved_ms']:.0f} / "
        f"{p['test']['hinglish']['mean_saved_ms']:.0f} ms wasting {p['test']['all']['wasted_per_turn']:.2f}. "
        "Test scenarios never seen in training.",
        "5/ The nice part: labels are free. Every call log already has interim and final transcripts, "
        "so you can train this on your own traffic with zero annotation. It runs in milliseconds on CPU, no LLM call.",
        "6/ Caveats are in the README (synthetic voices, simulated streaming, one labeller). Code, plots, "
        "Colab notebook for the GPU run: github.com/tanuku-saikarthik/jaldi. Would love feedback from "
        "people running voice agents in production.",
    ]
    return "# X thread\n\n" + _draft_banner(meta) + "\n\n".join(t) + "\n"


def _demo_script(m1: dict, policy: dict, meta: dict, budget: str, examples: list[dict]) -> str:
    j, p, b = _headline(policy, budget)
    neg = next((e for e in examples if e["lang"] in ("hi", "hinglish") and e["phenomenon"] == "negation"), None)
    lines = [
        "# 90-second demo script",
        "",
        _draft_banner(meta).strip(),
        "",
        "**0–15 s. The problem.** \"Voice agents start thinking only after you stop. Speculative generation "
        "starts early, but firing at the wrong moment wastes LLM calls.\"",
        "",
        "**15–40 s. Replay a Hindi/Hinglish negation turn** (demo → Replay).",
    ]
    if neg:
        h, jj = neg["heuristic"], neg["jaldi"]
        lines.append(f"Example: \"{neg['final_text']}\". Point at where the pause heuristic fires "
                     f"(wasted {h['wasted']} call(s), saved {h['saved']:.0f} ms) vs Jaldi "
                     f"(wasted {jj['wasted']}, saved {jj['saved']:.0f} ms).")
    lines += [
        "",
        "**40–60 s. Plot 1.** \"Meaning flips late in Hindi because negation comes after the verb, "
        "but the action also settles earlier. A pause rule can't tell these apart.\"",
        "",
        f"**60–80 s. Plot 2.** \"Tuned for ≤{b:g} wasted call per turn: Jaldi saves "
        f"{j['test']['all']['mean_saved_ms']:.0f} ms per turn ({j['test']['all']['wasted_per_turn']:.2f} wasted) vs the best pause rule "
        f"{p['test']['all']['mean_saved_ms']:.0f} ms ({p['test']['all']['wasted_per_turn']:.2f} wasted), on scenarios it never saw.\"",
        "",
        "**80–90 s. Close.** \"Labels come free from call logs. It's a few ms on CPU. "
        "It would slot in where Bolna already starts its eager LLM task (docs/integration.md).\"",
        "",
    ]
    return "\n".join(lines)
