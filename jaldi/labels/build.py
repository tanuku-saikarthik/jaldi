"""Stage 4: attach L1 / L2 labels to every partial."""

from __future__ import annotations

from collections import Counter

from ..io import read_jsonl, write_jsonl
from . import rule_extractor as R
from .action_match import Extractor
from .transcript_match import l1_labels


def build_labels(corpus_path: str, partials_path: str, out_path: str, extractor: Extractor) -> dict:
    corpus = {r["id"]: r for r in read_jsonl(corpus_path)}
    out, gold_ok = [], Counter()
    for t in read_jsonl(partials_path):
        c = corpus[t["id"]]
        sc = c["scenario"]
        final_action = extractor(t["final_text"], sc)
        ok = R.same_action(final_action, c["gold_action"])
        gold_ok[(c["lang"], ok)] += 1
        texts = [p["text"] for p in t["partials"]]
        l1 = l1_labels(texts, t["final_text"])
        parts = []
        for p, y1 in zip(t["partials"], l1):
            a = extractor(p["text"], sc)
            parts.append({
                **p,
                "action": a,
                "actionable": int(R.is_actionable(a, sc)),
                "contradicts": int(R.contradicts(a, final_action)),
                "l1": y1,
                "l2": int(R.same_action(a, final_action)),
            })
        out.append({
            "id": c["id"], "lang": c["lang"], "split": c["split"], "scenario": sc,
            "phenomenon": c["phenomenon"], "correction_placement": c.get("correction_placement"),
            "speaker": c["speaker"], "gold_action": c["gold_action"],
            "final_text": t["final_text"], "final_action": final_action,
            "final_matches_gold": int(ok), "eot_ms": t["eot_ms"],
            "speech_end_ms": t["speech_end_ms"], "mode": t.get("mode", "?"), "partials": parts,
        })
    write_jsonl(out_path, out)
    langs = sorted({k[0] for k in gold_ok})
    acc = {lg: gold_ok[(lg, True)] / max(1, gold_ok[(lg, True)] + gold_ok[(lg, False)]) for lg in langs}
    total = sum(v for (lg, ok), v in gold_ok.items() if ok) / max(1, sum(gold_ok.values()))
    return {"extractor": extractor.kind, "final_action_matches_gold": round(total, 4),
            "by_lang": {k: round(v, 4) for k, v in acc.items()}, "n_turns": len(out)}
