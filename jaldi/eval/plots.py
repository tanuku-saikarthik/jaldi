"""Static figures for the README. Every value plotted is read from results/*.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (first three slots pass all-pairs CVD checks on light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LANG_COLOR = {"en": "#2a78d6", "hi": "#eb6834", "hinglish": "#1baf7a"}
LANG_NAME = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
FAM_STYLE = {
    "jaldi": ("#2a78d6", "o", "Jaldi (learned)"),
    "pause": ("#eb6834", "s", "pause ≥ X ms"),
    "stable": ("#1baf7a", "^", "stable for k partials"),
    "turn_detector": ("#eda100", "D", "turn detector ≥ τ"),
}
LANGS = ("en", "hi", "hinglish")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(nrows, ncols, w, h):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_m1(m1: dict, path: str) -> None:
    fig, axes = _fig(2, 2, 11, 7.5)
    bl = m1["by_lang"]

    ax = axes[0, 0]
    for i, lg in enumerate(LANGS):
        p, lo, hi = bl[lg]["meaning_flip_rate"]
        ax.bar(i, p * 100, width=0.6, color=LANG_COLOR[lg])
        ax.errorbar(i, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]], color=INK2, capsize=4, lw=1.2)
        ax.text(i, hi * 100 + 0.8, f"{p * 100:.1f}%", ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(3), [LANG_NAME[lg] for lg in LANGS])
    ax.set_ylabel("% of turns", color=INK2)
    ax.set_title("A. Late meaning flips\n(an actionable partial contradicts the final action; 95% CI)",
                 fontsize=10, color=INK, loc="left")

    ax = axes[0, 1]
    phens = ["negation", "late_correction_in_place", "late_correction_end"]
    names = ["negation", "correction\n(in place)", "correction\n(at end)"]
    w = 0.26
    for j, lg in enumerate(LANGS):
        vals = [m1["flip_by_phenomenon"][lg].get(ph, {"rate": 0})["rate"] * 100 for ph in phens]
        xs = [k + (j - 1) * w for k in range(len(phens))]
        ax.bar(xs, vals, width=w - 0.02, color=LANG_COLOR[lg], label=LANG_NAME[lg])
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(phens)), names)
    ax.set_ylim(0, 135)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% of those turns that flip", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    ax.set_title("B. Which phenomena cause flips", fontsize=10, color=INK, loc="left")

    ax = axes[1, 0]
    ends = []
    for lg in LANGS:
        sw = m1["pause_sweep_test_l2"][lg]
        xs = [s["x_ms"] for s in sw]
        ys = [s["wasted_per_turn"] for s in sw]
        ax.plot(xs, ys, color=LANG_COLOR[lg], lw=2, marker="o", ms=5, label=LANG_NAME[lg])
        ends.append([ys[-1], lg, xs[-1]])
    # direct labels at the right end, nudged apart so they never collide
    ends.sort()
    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    gap = max(0.07, span / 6)
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, lg, x in ends:
        ax.annotate(LANG_NAME[lg], (x, y), xytext=(8, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ax.set_xlim(right=max(e[2] for e in ends) * 1.12)
    ax.set_xlabel("pause threshold X (ms)", color=INK2)
    ax.set_ylabel("wasted LLM calls per turn", color=INK2)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C. Pause heuristic waste by language (test split, L2)", fontsize=10, color=INK, loc="left")

    ax = axes[1, 1]
    for i, lg in enumerate(LANGS):
        v = bl[lg]["l2_headroom_ms_median"]
        ax.bar(i, v, width=0.6, color=LANG_COLOR[lg])
        ax.text(i, v + 15, f"{v:.0f} ms", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3), [LANG_NAME[lg] for lg in LANGS])
    ax.set_ylabel("median ms before end of turn", color=INK2)
    ax.set_title("D. How early the action settles (oracle L2 headroom)", fontsize=10, color=INK, loc="left")
    _save(fig, path)


def _pareto_front(pts):
    pts = sorted(pts, key=lambda p: (p[0], -p[1]))
    front, best = [], -1
    for x, y in pts:
        if y > best:
            front.append((x, y))
            best = y
    return front


def plot_pareto(policy: dict, mode: str, path: str) -> None:
    pts = policy["points"][mode]
    fig, axes = _fig(1, 3, 13, 4.3)
    xmax = max(p["wasted_per_turn"] for p in pts if p["family"] != "every" and p["set"].startswith("test_"))
    for ax, lg in zip(axes[0], LANGS):
        sub = [p for p in pts if p["set"] == f"test_{lg}"]
        for fam, (col, mk, name) in FAM_STYLE.items():
            fp = sorted([p for p in sub if p["family"] == fam], key=lambda p: p["wasted_per_turn"])
            if not fp:
                continue
            xs = [p["wasted_per_turn"] for p in fp]
            ys = [p["mean_saved_ms"] for p in fp]
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=6, label=name, alpha=0.95)
        ev = next(p for p in sub if p["family"] == "every")
        ax.annotate(f"every partial: {ev['mean_saved_ms']:.0f} ms saved,\n{ev['wasted_per_turn']:.1f} wasted/turn",
                    xy=(0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=7.5, color=INK2)
        ax.scatter([0], [0], color=INK2, s=30, zorder=5)
        ax.set_xlim(-0.05, min(xmax, 3.0) * 1.05)
        ax.set_ylim(-10, policy["time_to_ready_ms"] * 1.02)
        ax.set_title(LANG_NAME[lg], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("wasted speculative LLM calls per turn  (● at origin = never speculate)",
                      color=INK2, fontsize=9)
    axes[0, 0].set_ylabel("mean latency saved per turn (ms)", color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, path)


def plot_cost(policy: dict, mode: str, path: str) -> None:
    cv = policy["cost_view"][mode]
    fig, axes = _fig(1, len(cv), 4.3 * len(cv), 4)
    for ax, (c, by_lang) in zip(axes[0], cv.items()):
        for lg in LANGS:
            curve = by_lang[lg]["jaldi_curve"]
            ax.plot([p["tau"] for p in curve], [p["cost"] for p in curve], color=LANG_COLOR[lg],
                    lw=2, label=LANG_NAME[lg])
            bt = by_lang[lg]["best_tau"]
            bc = by_lang[lg]["best_jaldi_cost"]
            ax.scatter([bt], [bc], color=LANG_COLOR[lg], s=50, zorder=5, edgecolor=SURFACE, linewidth=2)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.18 * (hi - lo))
        best = " · ".join(f"{LANG_NAME[lg]} {by_lang[lg]['best_tau']:g}" for lg in LANGS)
        ax.text(0.02, 0.98, f"lowest-cost τ: {best}", transform=ax.transAxes, fontsize=7.5,
                color=INK2, va="top")
        ax.set_title(f"one wasted call costs {float(c):.0f} ms", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Jaldi threshold τ", color=INK2)
    axes[0, 0].set_ylabel("cost per turn = c·waste − ms saved", color=INK2)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="center right")
    _save(fig, path)


def plot_reliability(model: dict, path: str) -> None:
    fig, axes = _fig(1, 2, 9, 4)
    for ax, lab in zip(axes[0], ("l1", "l2")):
        r = model["labels"][lab]["reliability_test"]
        pts = [(p, f) for p, f, n in zip(r["bin_mean_p"], r["bin_frac_pos"], r["bin_n"]) if n > 0]
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
        ax.plot([p for p, _ in pts], [f for _, f in pts], color=LANG_COLOR["en"], lw=2, marker="o", ms=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        e = model["labels"][lab]["ece_test"]
        ax.set_title(f"{lab.upper()} — ECE {e:.3f} (test)", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("predicted P(safe)", color=INK2)
    axes[0, 0].set_ylabel("observed fraction safe", color=INK2)
    _save(fig, path)
