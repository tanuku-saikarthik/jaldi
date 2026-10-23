"""Deterministic action extractor: text (any script / code-mixed) -> {intent, polarity, slots}.

Used for the CPU smoke run and as a cross-check against the LLM labeller. It only sees the text it
is given, so running it on a partial transcript never reads the future.
"""

from __future__ import annotations

from functools import lru_cache

from ..corpus import lexicon as L
from ..text import tokens

_ARTICLES = {"a", "an", "the", "my"}

NEG_ALWAYS = {"मत", "mat", "don't", "dont"}
NEG_NAHI = {"नहीं", "nahi", "not"}
LIGHT_AUX = {"करनी", "करना", "है", "कीजिए", "हैं", "karni", "karna", "hai", "kijiye", "hain"}

QUALIFIER_KEYS = {
    "window": ["window", "खिडकी"],
    "online": ["online", "ऑनलाइन"],
    "wallet": ["wallet", "वॉलेट"],
    "yes": ["door", "दरवाजे"],
    "ac": ["ac", "एसी"],
    "suv": ["suv", "एसयूवी"],
}


def _norm_tokens(text: str) -> tuple[str, ...]:
    t = tokens(text)
    while t and t[0] in _ARTICLES:
        t = t[1:]
    return tuple(t)


@lru_cache(maxsize=1)
def _verb_index() -> dict[str, str]:
    idx = {}
    for key, forms in L.VERBS.items():
        for f in forms.values():
            idx[tokens(f)[0]] = key
    return idx


@lru_cache(maxsize=1)
def _slot_patterns() -> list[tuple[str, str, tuple[str, ...]]]:
    """(slot, value, token pattern). Number slots are anchored to their context word."""
    pats = []
    nums = L.SLOT_VALUES["party_size"]
    for v, forms in nums.items():
        for lang, ctx_after in (("en", "people"), ("hi", "लोगों"), ("hinglish", "logon")):
            pats.append(("party_size", v, (tokens(forms[lang])[0], ctx_after)))
    for v, forms in L.SLOT_VALUES["time"].items():
        pats.append(("time", v, ("at", tokens(forms["en"])[0])))
        pats.append(("time", v, (tokens(forms["hi"])[0], "बजे")))
        pats.append(("time", v, (tokens(forms["hinglish"])[0], "baje")))
    for slot, values in L.SLOT_VALUES.items():
        if slot in ("party_size", "time"):
            continue
        for v, forms in values.items():
            for f in set(forms.values()):
                pats.append((slot, v, _norm_tokens(f)))
    # longest patterns first so "the day after tomorrow" beats "tomorrow"
    pats.sort(key=lambda p: -len(p[2]))
    return pats


VERB_KEYWORDS = frozenset(_verb_index())
NUMBER_WORDS = frozenset(
    tokens(f)[0] for v in L.SLOT_VALUES["time"].values() for f in v.values()
) | frozenset(tokens(f)[0] for v in L.SLOT_VALUES["party_size"].values() for f in v.values())


def find_slots(toks: tuple[str, ...]) -> dict[str, str]:
    """Last mention wins (handles corrections). Overlapping shorter matches are skipped."""
    taken = [False] * len(toks)
    found: dict[str, tuple[int, str]] = {}
    for slot, value, pat in _slot_patterns():
        n = len(pat)
        if n == 0:
            continue
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == pat and not any(taken[i:i + n]):
                for j in range(i, i + n):
                    taken[j] = True
                if slot not in found or i > found[slot][0]:
                    found[slot] = (i, value)
    return {s: v for s, (_, v) in found.items()}


def find_polarity(toks: tuple[str, ...]) -> str:
    for i, t in enumerate(toks):
        if t in NEG_ALWAYS:
            return "neg"
        if t in NEG_NAHI:
            if t == "not":
                return "neg"
            prev_is_verb = i > 0 and toks[i - 1] in VERB_KEYWORDS
            next_is_aux = i + 1 < len(toks) and toks[i + 1] in LIGHT_AUX
            if prev_is_verb or next_is_aux:
                return "neg"
    return "pos"


def extract(text: str, scenario: str) -> dict:
    toks = tuple(tokens(text))
    verb = None
    for t in toks:
        if t in _verb_index():
            verb = _verb_index()[t]
    intent = None
    if verb is not None:
        for name, spec in L.SCENARIOS[scenario].items():
            if spec["verb"] == verb:
                intent = name
    slots = find_slots(toks)
    if intent is not None:
        allowed = L.schema_slots(scenario, intent)
        slots = {k: v for k, v in slots.items() if k in allowed}
        spec = L.intent_spec(scenario, intent)
        if "qualifier" in spec:
            q = spec["qualifier"]
            if any(k in toks for k in QUALIFIER_KEYS[q["value"]]):
                slots[q["slot"]] = q["value"]
    return {"intent": intent, "polarity": find_polarity(toks), "slots": dict(sorted(slots.items()))}


def is_actionable(action: dict, scenario: str) -> bool:
    """An agent would act on it: intent known and all required slots present."""
    if action.get("intent") is None:
        return False
    req = L.intent_spec(scenario, action["intent"])["required"]
    return all(r in action.get("slots", {}) for r in req)


def same_action(a: dict, b: dict) -> bool:
    return (a.get("intent") == b.get("intent") and a.get("polarity") == b.get("polarity")
            and dict(a.get("slots", {})) == dict(b.get("slots", {})))


def contradicts(partial: dict, final: dict) -> bool:
    """Meaning flip: intent or polarity differ, or a slot present in both has a different value.
    (Missing slots that arrive later are incompleteness, not contradiction.)"""
    if partial.get("intent") != final.get("intent") or partial.get("polarity") != final.get("polarity"):
        return True
    ps, fs = partial.get("slots", {}), final.get("slots", {})
    return any(k in fs and fs[k] != v for k, v in ps.items()) or any(k not in fs for k in ps)
