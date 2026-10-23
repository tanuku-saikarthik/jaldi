"""L2 label: the action derived from the partial equals the action derived from the final transcript.

Extractors:
  * "rule": deterministic parser (CPU smoke run; also a cross-check)
  * "llm":  actions precomputed on GPU by `label_with_vllm` and stored in data/llm_actions.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..corpus import lexicon as L
from ..io import read_jsonl, write_jsonl
from ..text import normalize
from . import rule_extractor as R


class Extractor:
    def __init__(self, kind: str = "rule", llm_file: str | Path | None = None):
        self.kind = kind
        self.table: dict[tuple[str, str], dict] = {}
        if kind == "llm":
            for r in read_jsonl(llm_file):
                self.table[(r["scenario"], normalize(r["text"]))] = r["action"]

    def __call__(self, text: str, scenario: str) -> dict:
        if self.kind == "rule":
            return R.extract(text, scenario)
        key = (scenario, normalize(text))
        if key not in self.table:
            raise KeyError(f"no LLM action for {key}; rerun the label stage on GPU")
        return self.table[key]


# ------------------------------------------------------------------ GPU: vLLM labeller
def action_json_schema(scenario: str) -> dict:
    intents = list(L.SCENARIOS[scenario])
    slot_props = {}
    for intent in intents:
        for s in L.schema_slots(scenario, intent):
            if s in L.SLOT_VALUES:
                slot_props[s] = {"type": ["string", "null"], "enum": sorted(L.SLOT_VALUES[s]) + [None]}
            else:
                q = L.intent_spec(scenario, intent)["qualifier"]
                slot_props[s] = {"type": ["string", "null"], "enum": [q["value"], None]}
    return {
        "type": "object",
        "properties": {
            "intent": {"type": ["string", "null"], "enum": intents + [None]},
            "polarity": {"type": "string", "enum": ["pos", "neg"]},
            "slots": {"type": "object", "properties": slot_props, "additionalProperties": False},
        },
        "required": ["intent", "polarity", "slots"],
        "additionalProperties": False,
    }


def _prompt(scenario: str, text: str) -> str:
    lines = []
    for intent, spec in L.SCENARIOS[scenario].items():
        slots = sorted(L.schema_slots(scenario, intent))
        lines.append(f"- {intent} (required: {spec['required'] or 'none'}; optional: {slots})")
    values = {s: sorted(v) for s, v in L.SLOT_VALUES.items()}
    return (
        "You are the understanding module of a phone voice agent. The caller may speak English, "
        "Hindi or Hinglish. The transcript may be INCOMPLETE (the caller is still talking).\n"
        f"Scenario: {scenario}. Possible intents:\n" + "\n".join(lines) + "\n"
        f"Allowed slot values: {json.dumps(values, ensure_ascii=False)}\n"
        "Return the action you would take RIGHT NOW from exactly this text, as JSON with keys "
        "intent (null if no intent is stated yet), polarity ('neg' if the caller says NOT to do it), "
        "slots (only slots actually mentioned; if the caller corrects a value, use the corrected one).\n"
        f"Transcript: {text}\nJSON:"
    )


def label_with_vllm(partials_path: str, corpus_path: str, out_path: str, model: str,
                    max_model_len: int = 4096) -> dict:
    """Run once on GPU. Extracts an action for every unique (scenario, partial text)."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    corpus = {r["id"]: r for r in read_jsonl(corpus_path)}
    uniq: dict[tuple[str, str], str] = {}
    for t in read_jsonl(partials_path):
        sc = corpus[t["id"]]["scenario"]
        for p in t["partials"] + [{"text": t["final_text"]}]:
            uniq.setdefault((sc, normalize(p["text"])), p["text"])
    keys = list(uniq)

    llm = LLM(model=model, max_model_len=max_model_len, seed=0)
    params_by_scenario = {}
    for sc in L.SCENARIOS:
        schema = action_json_schema(sc)
        params_by_scenario[sc] = _sampling_params(SamplingParams, schema)

    t0 = time.time()
    outs = []
    for sc in L.SCENARIOS:
        ks = [k for k in keys if k[0] == sc]
        if not ks:
            continue
        prompts = [_chat(llm, _prompt(sc, uniq[k])) for k in ks]
        res = llm.generate(prompts, params_by_scenario[sc])
        for k, r in zip(ks, res):
            outs.append((k, _parse(r.outputs[0].text)))
    rows = [{"scenario": k[0], "text": uniq[k], "action": a} for k, a in outs]
    write_jsonl(out_path, rows)
    return {"n_unique_texts": len(rows), "seconds": round(time.time() - t0, 1), "model": model}


def _sampling_params(SamplingParams, schema: dict):
    """Structured JSON output across vLLM versions (structured_outputs >= 0.10, guided_decoding before)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams  # noqa: PLC0415

        return SamplingParams(temperature=0, max_tokens=200,
                              structured_outputs=StructuredOutputsParams(json=schema))
    except (ImportError, TypeError):
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams  # noqa: PLC0415

        return SamplingParams(temperature=0, max_tokens=200,
                              guided_decoding=GuidedDecodingParams(json=schema))
    except (ImportError, TypeError):
        return SamplingParams(temperature=0, max_tokens=200)


def _chat(llm, prompt: str) -> str:
    tok = llm.get_tokenizer()
    msgs = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _parse(text: str) -> dict:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        a = json.loads(text[start:end])
    except ValueError:
        return {"intent": None, "polarity": "pos", "slots": {}}
    slots = {k: v for k, v in (a.get("slots") or {}).items() if v is not None}
    return {"intent": a.get("intent"), "polarity": a.get("polarity", "pos"),
            "slots": dict(sorted(slots.items()))}
