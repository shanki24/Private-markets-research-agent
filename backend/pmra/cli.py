"""Command line entry point.

  python -m pmra.cli ingest
  python -m pmra.cli ask "Compare Northvale and Ashgrove TVPI" --agent planner
  python -m pmra.cli extract --submit
  python -m pmra.cli review --reviewer alice
  python -m pmra.cli bench
  python -m pmra.cli export
Set PMRA_PROVIDER=anthropic (+ ANTHROPIC_API_KEY) for LLM-backed runs; default is offline.
"""
from __future__ import annotations

import argparse
import json

from . import config
from .approval import ApprovalQueue
from .ingest import ingest_dir, load_chunks
from .llm import LLM
from .rag import RAG
from .retriever import Retriever


def main():
    ap = argparse.ArgumentParser(prog="pmra")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--agent", choices=["rag", "planner", "refine"], default="planner")
    a.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    a.add_argument("--k", type=int, default=config.TOP_K)
    e = sub.add_parser("extract")
    e.add_argument("--doc")
    e.add_argument("--submit", action="store_true", help="send to the human approval queue")
    r = sub.add_parser("review")
    r.add_argument("--reviewer", required=True)
    sub.add_parser("approved")
    sub.add_parser("bench")
    sub.add_parser("export")
    sub.add_parser("registry")
    args = ap.parse_args()

    if args.cmd == "ingest":
        print(f"{len(ingest_dir())} chunks written to {config.CHUNKS_FILE}")
    elif args.cmd == "ask":
        from .agents.evaluator_optimizer import EvaluatorOptimizer
        from .agents.planner_executor import PlannerExecutor
        rag = RAG(Retriever(load_chunks(), args.retriever), LLM(), args.k)
        run = {"rag": rag.answer, "planner": PlannerExecutor(rag).run, "refine": EvaluatorOptimizer(rag).run}[args.agent]
        ans = run(args.question)
        print("\n" + ans.text + "\n")
        for c in ans.citations:
            print(f"  [{c.tag}] {c.title} / {c.section}  ({c.chunk_id})")
        u = ans.usage
        print(f"\n  {u.input_tokens} in / {u.output_tokens} out tokens, ${u.cost_usd:.5f}, {ans.latency_s*1000:.0f} ms")
        if ans.trace:
            print("  trace:", json.dumps(ans.trace, indent=2)[:2000])
    elif args.cmd == "extract":
        from .extraction import extract
        chunks, llm, q = load_chunks(), LLM(), ApprovalQueue()
        for d in sorted({c.doc_id for c in chunks}):
            if args.doc and d != args.doc:
                continue
            m, _ = extract([c for c in chunks if c.doc_id == d], llm)
            print(m.model_dump_json(indent=1) if not args.submit else f"{d}: queued {q.submit(m, llm.version_id)}")
    elif args.cmd == "review":
        ApprovalQueue().review_cli(args.reviewer)
    elif args.cmd == "approved":
        print(json.dumps(ApprovalQueue().approved(), indent=1))
    elif args.cmd == "bench":
        from .eval.benchmark import run_all
        res = run_all(LLM())
        for r_ in res["retrieval"]:
            print(f"{r_['retriever']:<52} recall@5 {r_['recall@5']:.3f}  MRR {r_['mrr']:.3f}  nDCG {r_['ndcg@5']:.3f}")
        for x in res["answers"]:
            print(f"{x['agent']:<8} acc {x['accuracy']:.3f}  cite-prec {x['citation_precision']:.3f}  "
                  f"p50 {x['latency_s_p50']*1000:.0f}ms  ${x['cost_usd_per_query']:.5f}/q")
        print("extraction field accuracy", round(res["extraction"]["field_accuracy"], 3))
    elif args.cmd == "export":
        from .export_web import export
        export()
    elif args.cmd == "registry":
        from .registry import Registry
        print(json.dumps(Registry().export(), indent=1))


if __name__ == "__main__":
    main()
