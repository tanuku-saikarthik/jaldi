from collections import Counter

from jaldi.config import Config
from jaldi.corpus import lexicon as L
from jaldi.corpus.generate import generate
from jaldi.labels import rule_extractor as R
from jaldi.labels.transcript_match import l1_labels


def test_corpus_gate():
    cfg = Config()
    rows = generate(cfg)
    assert len(rows) >= 1000
    assert all(not L.validate_action(r["scenario"], r["gold_action"]) for r in rows)
    # rule extractor recovers the gold action from every final transcript
    assert all(R.same_action(R.extract(r["final_text"], r["scenario"]), r["gold_action"]) for r in rows)


def test_splits_are_scenario_and_speaker_disjoint():
    cfg = Config()
    rows = generate(cfg)
    sc = {s: {r["scenario"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    assert not (sc["train"] & sc["test"]) and not (sc["train"] & sc["val"]) and not (sc["val"] & sc["test"])
    for lang in L.LANGS:
        spk = {s: {r["speaker"] for r in rows if r["split"] == s and r["lang"] == lang} for s in ("train", "test")}
        assert not (spk["train"] & spk["test"])


def test_phenomenon_rates_similar_across_languages():
    rows = generate(Config())
    for p in ("negation", "late_correction", "trailing_qualifier", "hesitation"):
        rates = []
        for lg in L.LANGS:
            sub = [r for r in rows if r["lang"] == lg]
            rates.append(sum(r["phenomenon"] == p for r in sub) / len(sub))
        assert max(rates) - min(rates) < 0.06, (p, rates)
    c = Counter(r["phenomenon"] for r in rows)
    assert 0.3 < 1 - c["plain"] / len(rows) < 0.5


def test_hindi_negation_flips_late_english_does_not():
    hi = R.extract("मुझे कल की बुकिंग कैंसल", "restaurant")
    hi_final = R.extract("मुझे कल की बुकिंग कैंसल नहीं करनी है", "restaurant")
    assert hi["polarity"] == "pos" and hi_final["polarity"] == "neg"
    assert R.contradicts(hi, hi_final)
    en = R.extract("I don't want to cancel", "restaurant")
    assert en["polarity"] == "neg"


def test_correction_marker_is_not_negation():
    a = R.extract("mujhe saat baje nahi aath baje chaar logon ke liye table book karni hai", "restaurant")
    assert a["polarity"] == "pos" and a["slots"]["time"] == "8" and a["slots"]["party_size"] == "4"


def test_hinglish_kar_do_is_not_number_two():
    a = R.extract("kal table book kar do", "restaurant")
    assert "party_size" not in a["slots"]


def test_l1_labels():
    assert l1_labels(["book a", "book a table.", "Book a table"], "book a table") == [0, 1, 1]
