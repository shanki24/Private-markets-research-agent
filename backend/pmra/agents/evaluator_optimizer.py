"""Evaluator-optimizer loop.

Generator drafts a cited answer. Evaluator scores it with deterministic checks
(citation coverage, numeric grounding, non-empty) plus an optional LLM judge.
Optimizer applies a fix strategy (query rewrite, wider retrieval, feedback-regeneration)
until the score clears the threshold or the iteration budget runs out. Best draft wins.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from ..llm import LLM, Usage
from ..rag import NUM, RAG, Answer, _sentences

JUDGE_SYSTEM = """You grade an analyst answer against its sources. Return ONLY JSON:
{"faithfulness": 0-1, "completeness": 0-1, "issues": [short strings]}
faithfulness: every claim supported by the cited sources. completeness: fully answers the question."""

SYNONYMS = {"irr": "internal rate of return net IRR", "tvpi": "total value to paid-in TVPI",
            "dpi": "distributions to paid-in DPI", "nav": "net asset value NAV",
            "carry": "carried interest", "fee": "management fee"}


@dataclass
class Evaluation:
    score: float
    citation_coverage: float
    numeric_grounding: float
    answered: float
    judge: dict | None
    issues: list[str]


def _norm_num(s: str) -> str:
    return re.sub(r"[,\s]", "", s.lower())


class Evaluator:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    def __call__(self, a: Answer, chunks_by_id: dict) -> tuple[Evaluation, Usage]:
        sents = [s for s in _sentences(a.text) if len(s.split()) > 2]
        cov = sum(bool(re.search(r"\[S\d+\]", s)) for s in sents) / max(len(sents), 1)
        cited_text = " ".join(chunks_by_id[c.chunk_id].text for c in a.citations if c.chunk_id in chunks_by_id)
        nums = [n.strip() for n in NUM.findall(re.sub(r"\[S\d+\]", "", a.text)) if re.search(r"\d", n)]
        nums = [n for n in nums if not re.fullmatch(r"\d{1,2}|20\d\d", n.strip())]  # ignore dates/counters
        grounded = sum(_norm_num(re.sub(r"[a-z%]+$", "", n.lower())) in _norm_num(cited_text) for n in nums)
        ground = grounded / len(nums) if nums else 1.0
        answered = 0.0 if "not found" in a.text.lower() or not a.text.strip() else 1.0
        issues = []
        if cov < 1:
            issues.append(f"{round((1-cov)*len(sents))} sentence(s) lack citations")
        if ground < 1:
            issues.append(f"{len(nums)-grounded} figure(s) not found in cited sources")
        if not answered:
            issues.append("no answer produced")
        judge, usage = None, Usage()
        if not self.llm.offline:
            try:
                text, usage = self.llm.complete(JUDGE_SYSTEM, f"Question: {a.question}\nAnswer: {a.text}\n\nCited sources:\n{cited_text}", 300)
                judge = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))
                issues += judge.get("issues", [])
            except Exception:
                judge = None
        parts = [cov, ground, answered] + ([judge["faithfulness"], judge["completeness"]] if judge else [])
        return Evaluation(sum(parts) / len(parts), cov, ground, answered, judge, issues), usage


class EvaluatorOptimizer:
    def __init__(self, rag: RAG, evaluator: Evaluator | None = None, threshold: float = 0.9, max_iters: int = 3):
        self.rag, self.threshold, self.max_iters = rag, threshold, max_iters
        self.evaluator = evaluator or Evaluator(LLM(rag.llm.provider, rag.llm.model))
        self.by_id = {c.chunk_id: c for c in rag.retriever.chunks}

    def rewrite(self, q: str) -> str:
        extra = [v for k, v in SYNONYMS.items() if k in q.lower()]
        return q + (" " + " ".join(extra) if extra else "")

    def run(self, question: str) -> Answer:
        t0 = time.perf_counter()
        usage, trace, best = Usage(), [], None
        query, k, feedback = question, self.rag.k, ""
        for it in range(1, self.max_iters + 1):
            a = self.rag.answer(query, k=k, feedback=feedback)
            a.question = question
            ev, u = self.evaluator(a, self.by_id)
            usage.add(a.usage).add(u)
            trace.append({"iteration": it, "query": query, "k": k, "draft": a.text, "score": round(ev.score, 3),
                          "citation_coverage": round(ev.citation_coverage, 3),
                          "numeric_grounding": round(ev.numeric_grounding, 3), "issues": ev.issues})
            if best is None or ev.score > best[0]:
                best = (ev.score, a)
            if ev.score >= self.threshold:
                break
            # optimizer: choose next strategy based on what failed
            if ev.answered == 0 or ev.numeric_grounding < 1:
                query, k = self.rewrite(question), k + 3
            feedback = "\n".join(ev.issues)
        a = best[1]
        a.usage, a.trace = usage, trace
        a.latency_s = usage.latency_s = time.perf_counter() - t0
        return a
