"""Benchmarks: retrieval quality, answer accuracy, citation quality, latency, tokens, cost, extraction.

Every run is logged to the registry against a model version and a dataset hash,
so results are comparable across retrievers, agents, providers and fine-tuned models.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path

from .. import config
from ..agents.evaluator_optimizer import EvaluatorOptimizer
from ..agents.planner_executor import PlannerExecutor
from ..extraction import METRIC_FIELDS, extract
from ..ingest import load_chunks
from ..llm import LLM
from ..rag import RAG
from ..registry import Registry, file_hash
from ..retriever import Retriever

GOLD_EXTRACT = config.ROOT / "data" / "eval" / "extraction_gold.json"


def load_eval(split: str = "test") -> list[dict]:
    rows = [json.loads(l) for l in config.EVAL_FILE.read_text().splitlines() if l.strip()]
    return [r for r in rows if split == "all" or r["split"] == split]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace(",", "")).strip()


def answer_correct(pred: str, gold: str) -> bool:
    p = _norm(re.sub(r"\[S\d+\]", "", pred))
    return all(_norm(g) in p for g in gold.split("|"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)] if xs else 0.0


# ---------------------------------------------------------------- retrieval
def bench_retrieval(kinds=("bm25", "dense", "hybrid"), k: int = 5, split="test") -> list[dict]:
    chunks, qs = load_chunks(), load_eval(split)
    results = []
    for kind in kinds:
        r = Retriever(chunks, kind)
        for routed in (False, True):
            rag = RAG(r, LLM("offline"), k)
            rec, mrr, ndcg, hit1, lat = [], [], [], [], []
            for q in qs:
                gold = {tuple(g) for g in q["gold"]}
                t0 = time.perf_counter()
                hits = rag.retrieve(q["question"], k) if routed else r.search(q["question"], k)
                lat.append(time.perf_counter() - t0)
                rel = [(h.chunk.doc_id, h.chunk.section) in gold for h in hits]
                found = {(h.chunk.doc_id, h.chunk.section) for h in hits} & gold
                rec.append(len(found) / len(gold))
                first = next((i for i, x in enumerate(rel) if x), None)
                mrr.append(1 / (first + 1) if first is not None else 0)
                hit1.append(1.0 if rel and rel[0] else 0.0)
                dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(rel) if x)
                idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
                ndcg.append(dcg / idcg)
            results.append({"retriever": r.name + (" +fund-routing" if routed else ""), "k": k,
                            f"recall@{k}": statistics.mean(rec), "mrr": statistics.mean(mrr),
                            f"ndcg@{k}": statistics.mean(ndcg), "hit@1": statistics.mean(hit1),
                            "latency_ms_p50": 1000 * pct(lat, .5), "latency_ms_p95": 1000 * pct(lat, .95)})
    return results


# ---------------------------------------------------------------- answers
def bench_answers(llm: LLM, agents=("rag", "planner", "refine"), split="test") -> list[dict]:
    chunks, qs = load_chunks(), load_eval(split)
    rag = RAG(Retriever(chunks, "hybrid"), llm)
    runners = {"rag": rag.answer, "planner": PlannerExecutor(rag).run, "refine": EvaluatorOptimizer(rag).run}
    out = []
    for name in agents:
        rows = []
        for q in qs:
            a = runners[name](q["question"])
            gold_docs = {g[0] for g in q["gold"]}
            cited_docs = [c.doc_id for c in a.citations]
            rows.append({
                "id": q["id"], "type": q["type"], "correct": answer_correct(a.text, q["answer"]),
                "cit_precision": sum(d in gold_docs for d in cited_docs) / len(cited_docs) if cited_docs else 0.0,
                "cit_recall": len(set(cited_docs) & gold_docs) / len(gold_docs),
                "latency_s": a.latency_s, "in_tok": a.usage.input_tokens, "out_tok": a.usage.output_tokens,
                "cost": a.usage.cost_usd, "calls": a.usage.calls, "answer": a.text,
            })
        n = len(rows)
        lat = [r["latency_s"] for r in rows]
        by_type = {t: statistics.mean(r["correct"] for r in rows if r["type"] == t)
                   for t in sorted({r["type"] for r in rows})}
        out.append({"agent": name, "model": llm.version_id, "n": n,
                    "accuracy": statistics.mean(r["correct"] for r in rows),
                    **{f"accuracy_{t}": v for t, v in by_type.items()},
                    "citation_precision": statistics.mean(r["cit_precision"] for r in rows),
                    "citation_recall": statistics.mean(r["cit_recall"] for r in rows),
                    "latency_s_p50": pct(lat, .5), "latency_s_p95": pct(lat, .95),
                    "tokens_in_avg": statistics.mean(r["in_tok"] for r in rows),
                    "tokens_out_avg": statistics.mean(r["out_tok"] for r in rows),
                    "llm_calls_avg": statistics.mean(r["calls"] for r in rows),
                    "cost_usd_per_query": statistics.mean(r["cost"] for r in rows),
                    "rows": rows})
    return out


# ---------------------------------------------------------------- extraction
def bench_extraction(llm: LLM) -> dict:
    chunks, gold = load_chunks(), json.loads(GOLD_EXTRACT.read_text())
    correct = total = 0
    per_field = {f: [0, 0] for f in METRIC_FIELDS}
    t0 = time.perf_counter()
    cost = 0.0
    for doc_id, g in gold.items():
        m, u = extract([c for c in chunks if c.doc_id == doc_id], llm)
        cost += u.cost_usd
        for f in METRIC_FIELDS:
            pred, want = getattr(m, f).value, g[f]
            ok = (pred is None and want is None) or (
                isinstance(pred, (int, float)) and want is not None and abs(pred - want) <= 1e-6 + 0.005 * abs(want))
            correct += ok
            total += 1
            per_field[f][0] += ok
            per_field[f][1] += 1
    return {"model": llm.version_id, "field_accuracy": correct / total,
            "per_field": {f: a / b for f, (a, b) in per_field.items()},
            "docs": len(gold), "latency_s_total": time.perf_counter() - t0, "cost_usd_total": cost}


def run_all(llm: LLM | None = None, out_dir: Path = config.ARTIFACTS) -> dict:
    llm = llm or LLM()
    reg = Registry()
    reg.register_model(llm.version_id, "base" if not llm.provider.startswith("hf:") else "local",
                       base=llm.model, params={"provider": llm.provider})
    dh = file_hash(config.EVAL_FILE)
    res = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset_hash": dh,
           "retrieval": bench_retrieval(), "answers": bench_answers(llm), "extraction": bench_extraction(llm)}
    for r in res["retrieval"]:
        reg.log_run(llm.version_id, "retrieval", {k: v for k, v in r.items() if k != "retriever"},
                    {"retriever": r["retriever"]}, dh)
    for a in res["answers"]:
        reg.log_run(llm.version_id, f"answers:{a['agent']}", {k: v for k, v in a.items() if k not in ("rows", "agent", "model")},
                    {"agent": a["agent"], "retriever": "hybrid"}, dh)
    reg.log_run(llm.version_id, "extraction", {k: v for k, v in res["extraction"].items() if isinstance(v, (int, float))},
                {"task": "fund_metrics"}, file_hash(GOLD_EXTRACT))
    path = out_dir / f"bench_{llm.version_id.replace('/', '_')}.json"
    path.write_text(json.dumps(res, indent=2))
    return res
