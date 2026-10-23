"""Text cues from the partial transcript at time t (no lookahead).

Cue lists are generic function words for English / Hindi / romanized Hindi, not scenario words.
Verb keywords are the four generic action verbs used across scenarios.
"""

from __future__ import annotations

from ..labels.rule_extractor import NUMBER_WORDS, VERB_KEYWORDS
from ..text import codemix_index, is_devanagari, tokens

NEG = {"नहीं", "nahi", "मत", "mat", "don't", "dont", "not"}
END_AUX = {"है", "हैं", "hai", "hain"}
LIGHT = {"करनी", "करना", "कीजिए", "दीजिए", "कर", "सकते", "देना", "karni", "karna", "kijiye",
         "dijiye", "kar", "sakte", "dena"}
FUNCTION = {
    "के", "की", "का", "को", "में", "से", "पर", "पे", "लिए", "वाला", "वाली", "साथ",
    "ke", "ki", "ka", "ko", "mein", "se", "pe", "par", "liye", "wala", "wali", "saath",
    "for", "at", "to", "with", "by", "the", "a", "an", "my", "of", "in", "on", "it",
}
CONJ = {"and", "but", "or", "aur", "और", "lekin", "लेकिन", "ya", "या"}
FILLER = {"um", "uh", "so", "okay", "hi", "achha", "अच्छा", "haan", "हाँ", "ji", "जी", "suniye",
          "सुनिए", "matlab", "मतलब", "wait", "actually", "sorry", "mean"}
QUESTION = {"can", "could", "kya", "क्या"}

TEXT_FEATURES = [
    "n_words", "last_is_verb", "last_is_light", "last_is_end_aux", "last_is_neg",
    "last_is_function", "last_is_conj", "last_is_filler", "last_is_number", "has_verb", "has_neg",
    "has_end_aux", "tokens_since_verb", "has_question_word", "last_tok_chars", "last_in_vocab",
    "codemix", "frac_devanagari",
]

_VOCAB = VERB_KEYWORDS | NUMBER_WORDS | NEG | END_AUX | LIGHT | FUNCTION | CONJ | FILLER | QUESTION


def text_features(text: str) -> dict:
    toks = tokens(text)
    last = toks[-1] if toks else ""
    verb_pos = max((i for i, t in enumerate(toks) if t in VERB_KEYWORDS), default=-1)
    return {
        "n_words": len(toks),
        "last_is_verb": int(last in VERB_KEYWORDS),
        "last_is_light": int(last in LIGHT),
        "last_is_end_aux": int(last in END_AUX),
        "last_is_neg": int(last in NEG),
        "last_is_function": int(last in FUNCTION),
        "last_is_conj": int(last in CONJ),
        "last_is_filler": int(last in FILLER),
        "last_is_number": int(last in NUMBER_WORDS),
        "has_verb": int(verb_pos >= 0),
        "has_neg": int(any(t in NEG for t in toks)),
        "has_end_aux": int(any(t in END_AUX for t in toks)),
        "tokens_since_verb": (len(toks) - 1 - verb_pos) if verb_pos >= 0 else -1,
        "has_question_word": int(any(t in QUESTION for t in toks)),
        "last_tok_chars": len(last),
        "last_in_vocab": int(last in _VOCAB),
        "codemix": codemix_index(text),
        "frac_devanagari": (sum(is_devanagari(t) for t in toks) / len(toks)) if toks else 0.0,
    }
