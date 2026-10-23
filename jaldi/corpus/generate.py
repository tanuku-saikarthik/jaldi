"""Stage 1: template corpus of user turns, rendered as phrase chunks.

The same generative process runs for every language (same scenarios, same phenomenon rates, same
correction-placement mix). Only grammar decides where words land, so any difference between
languages downstream comes from word order, not from how the corpus was sampled.
"""

from __future__ import annotations

import random
import re

from ..config import Config
from . import lexicon as L

_GROUP = re.compile(r"\[([^\]]*)\]")
_SLOT = re.compile(r"\{(\w+)\}")
NEG_WORDS = {"नहीं", "nahi", "मत", "mat", "don't"}


def surface(slot: str, value: str, lang: str) -> str:
    return L.SLOT_VALUES[slot][value][lang]


def fill(template: str, lang: str, slots: dict) -> str:
    """Fill {slot} placeholders; drop [ ... ] groups whose slots are absent."""

    def keep_group(m: re.Match) -> str:
        inner = m.group(1)
        return inner if all(n in slots for n in _SLOT.findall(inner)) else ""

    t = _GROUP.sub(keep_group, template)
    t = _SLOT.sub(lambda m: surface(m.group(1), slots[m.group(1)], lang), t)
    return " ".join(t.split())


def obj_slots(spec: dict, lang: str) -> set[str]:
    return set(_SLOT.findall(spec["obj"][lang]))


def render(scenario: str, intent: str, lang: str, polarity: str, slots: dict,
           frame_idx: int) -> list[dict]:
    """Render the main sentence as chunks: [{text, kind, slots}]."""
    spec = L.intent_spec(scenario, intent)
    frames = L.FRAMES[lang][polarity]
    frame = frames[frame_idx % len(frames)]
    verb = L.VERBS[spec["verb"]][lang]
    light = L.LIGHT_VERB.get(lang, {}).get(spec["gender"], "")
    in_obj = obj_slots(spec, lang)
    obj_text = fill(spec["obj"][lang], lang, slots)

    adj_chunks = []
    for s in ("day", "time"):
        if s in slots and s not in in_obj:
            adj_chunks.append({
                "text": L.ADJUNCT[lang][s].format(v=surface(s, slots[s], lang)),
                "kind": "adj", "slots": [s],
            })

    chunks = []
    for part in frame:
        if part == "{O}":
            chunks.append({"text": obj_text, "kind": "obj",
                           "slots": sorted(in_obj & set(slots))})
        elif part == "{A}":
            chunks.extend(adj_chunks)
        else:
            text = " ".join(part.replace("{V}", verb).replace("{K}", light).split())
            has_neg = any(w in NEG_WORDS for w in text.split())
            kind = "verb" if "{V}" in part else ("neg" if has_neg else "frame")
            chunks.append({"text": text, "kind": kind, "slots": []})
    return [c for c in chunks if c["text"]]


def _eligible(spec: dict, phen: str, slots: dict) -> bool:
    if phen == "negation":
        return spec["negatable"]
    if phen == "trailing_qualifier":
        return "qualifier" in spec
    if phen in ("hesitation", "late_correction"):
        return len(slots) > 0
    return True


def make_turn(rng: random.Random, cfg: Config, idx: int, scenario: str, intent: str, lang: str,
              phen: str) -> dict | None:
    spec = L.intent_spec(scenario, intent)
    slots = {s: rng.choice(sorted(L.SLOT_VALUES[s])) for s in spec["required"]}
    for s in spec["optional"]:
        if rng.random() < 0.5:
            slots[s] = rng.choice(sorted(L.SLOT_VALUES[s]))
    if not _eligible(spec, phen, slots):
        return None

    polarity = "neg" if phen == "negation" else "pos"
    gold = {"intent": intent, "polarity": polarity, "slots": dict(slots)}
    frame_idx = rng.randrange(10)
    boundary = {}  # chunk index -> forced pause type before that chunk

    if phen in ("plain", "negation"):
        chunks = render(scenario, intent, lang, polarity, slots, frame_idx)

    elif phen == "trailing_qualifier":
        q = spec["qualifier"]
        chunks = render(scenario, intent, lang, polarity, slots, frame_idx)
        boundary[len(chunks)] = rng.choice(["short", "long"])
        chunks.append({"text": q["text"][lang], "kind": "qual", "slots": [q["slot"]]})
        gold["slots"][q["slot"]] = q["value"]

    elif phen == "hesitation":
        s = rng.choice(sorted(slots))
        main = {k: v for k, v in slots.items() if k != s}
        chunks = render(scenario, intent, lang, polarity, main, frame_idx)
        boundary[len(chunks)] = "long"
        chunks.append({"text": L.AFTERTHOUGHT[s][lang].format(v=surface(s, slots[s], lang)),
                       "kind": "after", "slots": [s]})

    elif phen == "late_correction":
        s = rng.choice(sorted(slots))
        wrong = rng.choice(sorted(v for v in L.SLOT_VALUES[s] if v != slots[s]))
        chunks = render(scenario, intent, lang, polarity, {**slots, s: wrong}, frame_idx)
        corr = {
            "text": rng.choice(L.CORRECTION_MARKERS[lang]) + " "
            + L.AFTERTHOUGHT[s][lang].format(v=surface(s, slots[s], lang)),
            "kind": "corr", "slots": [s],
        }
        placement = rng.choice(["in_place", "end"])
        if placement == "in_place":
            pos = next(i for i, c in enumerate(chunks) if s in c["slots"]) + 1
        else:
            pos = len(chunks)
        chunks.insert(pos, corr)
        boundary[pos] = "short"
        gold["correction_placement"] = placement
    else:
        raise ValueError(phen)

    if rng.random() < 0.3:
        chunks.insert(0, {"text": rng.choice(L.FILLERS[lang]), "kind": "filler", "slots": []})
        boundary = {k + 1: v for k, v in boundary.items()}

    correction_placement = gold.pop("correction_placement", None)
    return {
        "id": f"t{idx:05d}",
        "scenario": scenario,
        "lang": lang,
        "segments": [c["text"] for c in chunks],
        "segment_kinds": [c["kind"] for c in chunks],
        "forced_pause_before": {str(k): v for k, v in boundary.items()},
        "final_text": " ".join(c["text"] for c in chunks),
        "phenomenon": phen,
        "correction_placement": correction_placement,
        "gold_action": gold,
    }


def split_of(scenario: str, cfg: Config) -> str:
    if scenario in cfg.test_scenarios:
        return "test"
    if scenario in cfg.val_scenarios:
        return "val"
    return "train"


def assign_speaker(rng: random.Random, lang: str, split: str) -> str:
    pool = L.SPEAKERS[lang]["test" if split == "test" else "train"]
    return rng.choice(pool)


def generate(cfg: Config) -> list[dict]:
    rng = random.Random(cfg.seed)
    intents = L.all_intents()
    langs, lw = zip(*cfg.lang_mix.items())
    phens, pw = zip(*cfg.phenomenon_mix.items())
    rows, idx = [], 0
    while len(rows) < cfg.n_turns:
        # Draw language and phenomenon first and keep them, so phenomenon rates are identical
        # across languages; only the intent/slots are redrawn until the phenomenon is possible.
        lang = rng.choices(langs, lw)[0]
        phen = rng.choices(phens, pw)[0]
        turn = None
        for _ in range(100):
            scenario, intent = rng.choice(intents)
            turn = make_turn(rng, cfg, idx, scenario, intent, lang, phen)
            if turn is not None:
                break
        idx += 1
        if turn is None:
            raise RuntimeError(f"could not realise phenomenon {phen}")
        turn["split"] = split_of(scenario, cfg)
        turn["speaker"] = assign_speaker(rng, lang, turn["split"])
        rows.append(turn)
    return rows


def hinglish_mixed_script(text: str) -> str:
    """Romanized Hinglish -> Hindi words in Devanagari, English words kept in Latin (TTS script check)."""
    out = []
    for tok in text.split():
        core = tok.strip(",.")
        tail = tok[len(core):]
        out.append(L.HINGLISH_TO_DEVANAGARI.get(core.lower(), core) + tail)
    return " ".join(out)
