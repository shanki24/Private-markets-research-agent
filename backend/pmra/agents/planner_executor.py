"""Planner-executor agent.

The planner turns a research question into a typed plan of tool calls; the executor runs
them in order, tracks evidence as numbered sources, and a synthesizer writes the cited answer.
Comparisons are computed by code from extracted figures, not by the LLM, so they're exact.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from ..extraction import METRIC_FIELDS, extract
from ..ingest import Chunk
from ..llm import LLM, Usage
from ..rag import RAG, Answer, Citation, resolve_citations
from ..retriever import Hit, resolve_funds, tokenize

METRIC_WORDS = {
    "tvpi_x": ["tvpi", "total value"], "dpi_x": ["dpi", "distributed to paid"], "rvpi_x": ["rvpi"],
    "net_irr_pct": ["net irr", "irr"], "gross_irr_pct": ["gross irr"], "nav_m": ["nav", "net asset value"],
    "commitments_m": ["commitments", "fund size"], "called_m": ["called"],
    "mgmt_fee_pct": ["management fee"], "carry_pct": ["carry", "carried interest"],
}
LOWER_IS_BETTER = {"mgmt_fee_pct", "carry_pct"}
LABELS = {"tvpi_x": "net TVPI", "dpi_x": "DPI", "rvpi_x": "RVPI", "net_irr_pct": "net IRR",
          "gross_irr_pct": "gross IRR", "nav_m": "NAV", "commitments_m": "commitments",
          "called_m": "called capital", "mgmt_fee_pct": "management fee", "carry_pct": "carried interest / incentive fee"}

PLANNER_SYSTEM = """You plan research over private-market fund reports. Return ONLY JSON:
{"steps": [{"id": "s1", "tool": "search_answer", "query": str, "fund": doc_id|null}
         | {"id": "s2", "tool": "extract_metric", "fund": doc_id, "metric": one of METRICS}
         | {"id": "s3", "tool": "compare", "inputs": [step ids of extract_metric], "metric": str}]}
Use extract_metric + compare for any numeric comparison or ranking across funds.
Use search_answer for qualitative or single-fund questions. Max 8 steps."""


@dataclass
class Step:
    id: str
    tool: str
    query: str | None = None
    fund: str | None = None
    metric: str | None = None
    inputs: list[str] = field(default_factory=list)
    output: dict = field(default_factory=dict)
    latency_s: float = 0.0


def detect_metric(question: str) -> str | None:
    q = question.lower()
    for m, words in sorted(METRIC_WORDS.items(), key=lambda kv: -max(len(w) for w in kv[1])):
        if any(w in q for w in words):
            return m
    return None


class PlannerExecutor:
    def __init__(self, rag: RAG, llm: LLM | None = None):
        self.rag, self.llm = rag, llm or rag.llm
        self.chunks: list[Chunk] = rag.retriever.chunks
        self.funds = {c.doc_id: c.meta.get("fund", c.title) for c in self.chunks}
        self._extract_cache: dict[str, object] = {}

    # ---------------- planning ----------------
    def plan(self, question: str) -> tuple[list[Step], Usage]:
        if not self.llm.offline:
            ctx = json.dumps({"FUNDS": self.funds, "METRICS": METRIC_FIELDS})
            try:
                text, u = self.llm.complete(PLANNER_SYSTEM, f"{ctx}\n\nQuestion: {question}", 600)
                raw = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))
                steps = [Step(**{k: v for k, v in s.items() if k in Step.__dataclass_fields__}) for s in raw["steps"]]
                if steps:
                    return steps, u
            except Exception:
                pass  # fall back to the deterministic planner
        return self._heuristic_plan(question), Usage()

    def _heuristic_plan(self, question: str) -> list[Step]:
        funds = resolve_funds(question, self.chunks)
        metric = detect_metric(question)
        if len(funds) >= 2 and metric:
            steps = [Step(f"s{i+1}", "extract_metric", fund=f, metric=metric) for i, f in enumerate(funds)]
            steps.append(Step(f"s{len(steps)+1}", "compare", metric=metric, inputs=[s.id for s in steps]))
            return steps
        if len(funds) >= 2:
            return [Step(f"s{i+1}", "search_answer", query=question, fund=f) for i, f in enumerate(funds)]
        return [Step("s1", "search_answer", query=question, fund=funds[0] if funds else None)]

    # ---------------- execution ----------------
    def run(self, question: str) -> Answer:
        t0 = time.perf_counter()
        steps, usage = self.plan(question)
        sources: list[Hit] = []

        def cite(chunk: Chunk) -> str:
            for i, h in enumerate(sources):
                if h.chunk.chunk_id == chunk.chunk_id:
                    return f"S{i+1}"
            sources.append(Hit(chunk, 0.0, len(sources) + 1))
            return f"S{len(sources)}"

        by_id = {}
        for s in steps:
            ts = time.perf_counter()
            if s.tool == "search_answer":
                q = s.query or question
                hits = (self.rag.retriever.search(q, 3, doc_filter=s.fund) if s.fund else self.rag.retrieve(q, 4))
                a = self.rag.answer(q, hits=hits)
                usage.add(a.usage)
                # re-map local [S#] tags to global ones
                text = re.sub(r"\[S(\d+)\]", lambda m: f"[{cite(hits[int(m.group(1))-1].chunk)}]"
                              if int(m.group(1)) <= len(hits) else "", a.text)
                s.output = {"answer": text}
            elif s.tool == "extract_metric":
                if s.fund not in self._extract_cache:
                    m, u = extract([c for c in self.chunks if c.doc_id == s.fund], self.llm)
                    usage.add(u)
                    self._extract_cache[s.fund] = m
                f = getattr(self._extract_cache[s.fund], s.metric)
                ch = next((c for c in self.chunks if c.chunk_id == f.evidence_chunk), None)
                s.output = {"fund": self.funds[s.fund], "value": f.value, "unit": f.unit,
                            "tag": cite(ch) if ch else None}
            elif s.tool == "compare":
                rows = [by_id[i].output for i in s.inputs if by_id[i].output.get("value") is not None]
                rev = s.metric not in LOWER_IS_BETTER
                rows.sort(key=lambda r: r["value"], reverse=rev)
                s.output = {"ranking": rows, "metric": s.metric}
            s.latency_s = time.perf_counter() - ts
            by_id[s.id] = s

        text, u = self._synthesize(question, steps, sources)
        usage.add(u)
        trace = [{"id": s.id, "tool": s.tool, "fund": s.fund, "metric": s.metric, "query": s.query,
                  "output": s.output, "latency_s": round(s.latency_s, 4)} for s in steps]
        dt = time.perf_counter() - t0
        usage.latency_s = dt
        return Answer(question, text, resolve_citations(text, sources), [h.chunk.chunk_id for h in sources],
                      usage, dt, trace)

    def _fmt(self, r: dict) -> str:
        v, u = r["value"], r["unit"]
        val = f"{v:g}x" if u == "x" else f"{v:g}%" if u == "%" else f"{v:,.0f}m"
        return f"{val} [{r['tag']}]" if r.get("tag") else val

    def _synthesize(self, question: str, steps: list[Step], sources: list[Hit]) -> tuple[str, Usage]:
        cmp = next((s for s in steps if s.tool == "compare"), None)
        if self.llm.offline or cmp:
            # deterministic synthesis: exact for computed comparisons
            if cmp and cmp.output["ranking"]:
                rk = cmp.output["ranking"]
                label = LABELS.get(cmp.metric, cmp.metric)
                lead = f"{rk[0]['fund']} has the {'lowest' if cmp.metric in LOWER_IS_BETTER else 'highest'} " \
                       f"{label} at {self._fmt(rk[0])}"
                rest = "; ".join(f"{r['fund']} {self._fmt(r)}" for r in rk[1:])
                return lead + (f", compared with {rest}." if rest else "."), Usage()
            return " ".join(s.output.get("answer", "") for s in steps if s.tool == "search_answer"), Usage()
        notes = "\n".join(f"- {s.id} {s.tool}: {json.dumps(s.output)}" for s in steps)
        srcs = "\n\n".join(f"[S{i+1}] ({h.chunk.title} / {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(sources))
        system = "Write a concise cited answer from the step results and sources. Cite with [S#]. Use only given facts."
        return self.llm.complete(system, f"Question: {question}\n\nStep results:\n{notes}\n\nSources:\n{srcs}", 500)
