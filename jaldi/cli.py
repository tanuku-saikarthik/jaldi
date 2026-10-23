"""jaldi CLI. CPU smoke run:  python -m jaldi.cli all

GPU stages (Colab): corpus -> tts -> partials --mode audio -> label --extractor llm -> measure-ttr
then features -> train -> simulate -> report (CPU is fine for these).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .config import Config
from .io import read_json, read_jsonl, write_json, write_jsonl


def _gate(name: str, ok: bool, detail: str) -> None:
    print(f"[gate] {name}: {'PASS' if ok else 'FAIL'} - {detail}")
    if not ok:
        print("Stopping. Log the failure in PROGRESS.md and fix before continuing.", file=sys.stderr)
        sys.exit(2)


def cmd_corpus(cfg: Config, args) -> None:
    from .corpus import lexicon as L
    from .corpus.generate import generate
    from .labels.rule_extractor import extract, same_action

    rows = generate(cfg)
    write_jsonl(cfg.data / "corpus.jsonl", rows)
    invalid = [r["id"] for r in rows if L.validate_action(r["scenario"], r["gold_action"])]
    rule_ok = sum(same_action(extract(r["final_text"], r["scenario"]), r["gold_action"]) for r in rows)
    counts = Counter((r["lang"], r["phenomenon"]) for r in rows)
    stats = {
        "n_turns": len(rows),
        "by_lang": dict(Counter(r["lang"] for r in rows)),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "phenomenon_by_lang": {lg: {p: counts[(lg, p)] for p in cfg.phenomenon_mix} for lg in cfg.lang_mix},
        "invalid_gold_actions": len(invalid),
        "rule_extractor_matches_gold_on_final_text": rule_ok / len(rows),
        "split_scenarios": {"val": cfg.val_scenarios, "test": cfg.test_scenarios},
    }
    write_json(cfg.results / "corpus_stats.json", stats)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    _gate("stage 1", len(rows) >= 1000 and not invalid,
          f"{len(rows)} turns, {len(invalid)} invalid gold actions")


def cmd_tts(cfg: Config, args) -> None:
    from .audio.assemble import assemble_all
    from .audio.tts import synthesize

    seg_dir, turn_dir = cfg.data / "audio" / "segments", cfg.data / "audio" / "turns"
    r = synthesize(str(cfg.data / "corpus.jsonl"), str(seg_dir), args.hinglish_script,
                   limit=args.limit, batch_size=args.batch_size, seed=cfg.seed)
    print(json.dumps(r, indent=1))
    if args.benchmark:
        return
    a = assemble_all(str(cfg.data / "corpus.jsonl"), str(seg_dir), str(turn_dir),
                     str(cfg.data / "timing.jsonl"), cfg, args.p_telephony)
    write_json(cfg.results / "tts_meta.json", {**r, **a})
    print(json.dumps(a, indent=1))


def cmd_partials(cfg: Config, args) -> None:
    if args.mode == "text_sim":
        from .asr.text_sim import simulate_turn

        recs = [simulate_turn(t, cfg) for t in read_jsonl(cfg.data / "corpus.jsonl")]
        write_jsonl(cfg.data / "partials.jsonl", recs)
        n = [len(r["partials"]) for r in recs]
        ok = all(r["partials"][-1]["text"] == r["final_text"] for r in recs)
        meta = {"mode": "text_sim", "turns": len(recs), "partials_per_turn_mean": sum(n) / len(n),
                "min": min(n), "max": max(n)}
        write_json(cfg.results / "partials_meta.json", meta)
        print(json.dumps(meta, indent=1))
        _gate("stage 3", ok and min(n) >= 3, "final partial equals final transcript; >=3 partials per turn")
        return
    from .asr.partials import audio_partials

    asr_by_lang = {"en": args.asr_en, "hi": args.asr_hi, "hinglish": args.asr_hinglish}
    r = audio_partials(str(cfg.data / "corpus.jsonl"), str(cfg.data / "timing.jsonl"),
                       str(cfg.data / "audio" / "turns"), str(cfg.data / "partials.jsonl"), cfg,
                       asr_by_lang, use_turn_detector=not args.no_turn_detector, limit=args.limit)
    write_json(cfg.results / "partials_meta.json", {"mode": "audio", "asr_by_lang": asr_by_lang, **r})
    print(json.dumps(r, indent=1))
    _gate("stage 3", r["final_partial_equals_full_clip"] >= 0.8,
          f"final partial == full-clip transcript on {r['final_partial_equals_full_clip']:.1%} of turns")


def cmd_label(cfg: Config, args) -> None:
    from .eval.m1 import m1_analysis
    from .labels.action_match import Extractor, label_with_vllm
    from .labels.build import build_labels

    llm_file = cfg.data / "llm_actions.jsonl"
    if args.extractor == "llm" and not llm_file.exists():
        r = label_with_vllm(str(cfg.data / "partials.jsonl"), str(cfg.data / "corpus.jsonl"),
                            str(llm_file), args.llm_model)
        print(json.dumps(r, indent=1))
    ex = Extractor(args.extractor, llm_file if args.extractor == "llm" else None)
    meta = build_labels(str(cfg.data / "corpus.jsonl"), str(cfg.data / "partials.jsonl"),
                        str(cfg.data / "labels.jsonl"), ex)
    meta["mode"] = read_json(cfg.results / "partials_meta.json")["mode"]
    write_json(cfg.results / "labels_meta.json", meta)
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    _gate("stage 4 (final action vs gold)", meta["final_action_matches_gold"] >= 0.9,
          f"{meta['final_action_matches_gold']:.1%} (needs >= 90%)")
    m1 = m1_analysis(read_jsonl(cfg.data / "labels.jsonl"), cfg)
    write_json(cfg.results / "m1.json", m1)
    print(json.dumps({k: {"meaning_flip": v["meaning_flip_rate"], "l2_headroom_median": v["l2_headroom_ms_median"]}
                      for k, v in m1["by_lang"].items()}, indent=1))


def cmd_features(cfg: Config, args) -> None:
    from .features.build import build_features

    df = build_features(read_jsonl(cfg.data / "labels.jsonl"))
    df.to_csv(cfg.data / "features.csv.gz", index=False)
    print(f"features: {df.shape[0]} rows x {df.shape[1]} cols")


def cmd_encoder(cfg: Config, args) -> None:
    import pandas as pd

    from .models.text_encoder import train_text_encoder

    turns = read_jsonl(cfg.data / "labels.jsonl")
    rows = pd.DataFrame([{"id": t["id"], "idx": i, "split": t["split"], "text": p["text"],
                          "l1": p["l1"], "l2": p["l2"]} for t in turns for i, p in enumerate(t["partials"])])
    train_text_encoder(rows, str(cfg.data / "enc_scores.csv"), args.encoder_model, args.epochs, cfg.seed)


def cmd_train(cfg: Config, args) -> None:
    import pandas as pd

    from .models.train import train_all

    df = pd.read_csv(cfg.data / "features.csv.gz")
    preds, out = train_all(df, cfg, str(cfg.data / "enc_scores.csv"))
    preds.to_csv(cfg.data / "preds.csv", index=False)
    write_json(cfg.results / "model.json", out)
    for lab, r in out["labels"].items():
        print(f"{lab}: AUROC {r['auroc_ci']}  ECE {r['ece_test']:.3f}  gate={r['gate']['passed']}")
    if not all(r["gate"]["passed"] for r in out["labels"].values()):
        print("[gate] stage 6: FAIL - model does not beat the best heuristic with non-overlapping CIs. "
              "Reported as is; analyze before claiming anything.")


def cmd_simulate(cfg: Config, args) -> None:
    import pandas as pd

    from .sim.baselines import run_sweep

    ttr_file = cfg.data / "ttr.json"
    if ttr_file.exists():
        cfg.time_to_ready_ms = float(read_json(ttr_file)["ttr_ms_p50"])
        print(f"using measured time-to-ready p50 = {cfg.time_to_ready_ms:.0f} ms")
    res = run_sweep(read_jsonl(cfg.data / "labels.jsonl"), pd.read_csv(cfg.data / "preds.csv"), cfg)
    res["time_to_ready_source"] = "measured (data/ttr.json)" if ttr_file.exists() else "assumed (config)"
    write_json(cfg.results / "policy.json", res)
    for lab, s in res["selected"]["l2"][str(cfg.waste_budgets[0])].items():
        if s.get("param") is not None:
            print(f"L2 budget {cfg.waste_budgets[0]} | {lab} (param {s['param']:g}): "
                  f"saved {s['test']['all']['mean_saved_ms']:.0f} ms, wasted {s['test']['all']['wasted_per_turn']:.2f}/turn")


def cmd_report(cfg: Config, args) -> None:
    from .eval.report import build_report

    s = build_report(cfg, args.readme)
    print("\n".join(s["verdicts"]))


def cmd_measure_ttr(cfg: Config, args) -> None:
    from .sim.ttr import measure_ttr

    r = measure_ttr(args.llm_model)
    write_json(cfg.data / "ttr.json", r)
    write_json(cfg.results / "ttr.json", r)
    print(json.dumps(r, indent=1))


def cmd_script_check(cfg: Config, args) -> None:
    from .asr.partials import script_check

    r = script_check(str(cfg.data / "corpus.jsonl"), str(cfg.data / "audio" / "script_check"), args.n)
    write_json(cfg.results / "script_check.json", r)
    print(json.dumps(r, indent=1))


def cmd_all(cfg: Config, args) -> None:
    args.mode, args.extractor = "text_sim", "rule"
    for f in (cmd_corpus, cmd_partials, cmd_label, cmd_features, cmd_train, cmd_simulate, cmd_report):
        print(f"\n===== {f.__name__.removeprefix('cmd_')} =====")
        f(cfg, args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="jaldi")
    ap.add_argument("--config", default=None, help="JSON overriding Config fields")
    ap.add_argument("--data-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus")
    p = sub.add_parser("tts")
    p.add_argument("--hinglish-script", choices=["roman", "mixed"], default="roman")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--benchmark", action="store_true", help="synthesize only (use with --limit 20)")
    p.add_argument("--p-telephony", type=float, default=0.5)
    p = sub.add_parser("script-check")
    p.add_argument("--n", type=int, default=10)
    p = sub.add_parser("partials")
    p.add_argument("--mode", choices=["text_sim", "audio"], default="text_sim")
    p.add_argument("--asr-en", default="whisper")
    p.add_argument("--asr-hi", default="whisper", choices=["whisper", "indicconformer"])
    p.add_argument("--asr-hinglish", default="whisper")
    p.add_argument("--no-turn-detector", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("label")
    p.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    sub.add_parser("features")
    p = sub.add_parser("encoder")
    p.add_argument("--encoder-model", default="google/muril-base-cased")
    p.add_argument("--epochs", type=int, default=2)
    sub.add_parser("train")
    sub.add_parser("simulate")
    p = sub.add_parser("report")
    p.add_argument("--readme", default="README.md")
    p = sub.add_parser("measure-ttr")
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    p = sub.add_parser("all", help="CPU smoke run: template corpus + simulated partials + rule labels")
    p.add_argument("--readme", default="README.md")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    cfg.data.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.results / "config.json")
    {
        "corpus": cmd_corpus, "tts": cmd_tts, "script-check": cmd_script_check,
        "partials": cmd_partials, "label": cmd_label, "features": cmd_features,
        "encoder": cmd_encoder, "train": cmd_train, "simulate": cmd_simulate,
        "report": cmd_report, "measure-ttr": cmd_measure_ttr, "all": cmd_all,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
