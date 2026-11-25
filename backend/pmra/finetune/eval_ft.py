"""Compare the base model with its QLoRA adapter on the held-out real reports.

Same prompts, same gold, same metrics: field accuracy, JSON validity, latency, tokens.
    python -m pmra.finetune.eval_ft --base Qwen/Qwen2.5-0.5B-Instruct --adapter artifacts/adapters/<run>
"""
from __future__ import annotations

import argparse
import json
import time

from .. import config
from ..eval.benchmark import GOLD_EXTRACT
from ..extraction import METRIC_FIELDS, extract
from ..ingest import load_chunks
from ..llm import LLM, Usage
from ..registry import Registry, file_hash


def score(llm: LLM) -> dict:
    chunks, gold = load_chunks(), json.loads(GOLD_EXTRACT.read_text())
    ok = total = valid = 0
    usage, lat = Usage(), []
    for doc_id, g in gold.items():
        t0 = time.perf_counter()
        try:
            m, u = extract([c for c in chunks if c.doc_id == doc_id], llm)
            usage.add(u)
            valid += 1
        except Exception:
            m = None
        lat.append(time.perf_counter() - t0)
        for f in METRIC_FIELDS:
            pred = getattr(m, f).value if m else None
            want = g[f]
            total += 1
            ok += (pred is None and want is None) or (isinstance(pred, (int, float)) and want is not None
                                                     and abs(pred - want) <= 0.005 * abs(want) + 1e-6)
    return {"field_accuracy": ok / total, "json_valid_rate": valid / len(gold),
            "latency_s_avg": sum(lat) / len(lat), "tokens_in": usage.input_tokens, "tokens_out": usage.output_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()
    reg, dh = Registry(), file_hash(GOLD_EXTRACT)
    results = {}
    for vid in (f"hf:{args.base}", f"hf:{args.base}@{args.adapter}"):
        llm = LLM(vid, args.base)
        reg.register_model(vid, "qlora" if "@" in vid else "local-base", base=args.base)
        results[vid] = score(llm)
        reg.log_run(vid, "extraction", results[vid], {"task": "fund_metrics", "held_out": True}, dh)
        del llm
    print(json.dumps(results, indent=2))
    (config.ARTIFACTS / "qlora_comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
