"""Retrieval-augmented answering with enforced, verifiable source citations."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .llm import LLM, Usage, estimate_tokens
from .retriever import Hit, Retriever, resolve_funds, tokenize

SYSTEM = """You are a private-markets research analyst. Answer ONLY from the numbered sources.
Rules:
- Cite every factual sentence with source tags like [S1] or [S2][S3].
- Quote figures exactly as written in the sources (units, x, %, currency).
- If the sources do not contain the answer, say "Not found in the provided reports."
- Be concise: 1-4 sentences."""

CITE = re.compile(r"\[S(\d+)\]")
NUM = re.compile(r"(?:EUR\s*|\$)?\d[\d,]*(?:\.\d+)?\s*(?:%|x|bps|basis points|MW|million|billion)?", re.I)


@dataclass
class Citation:
    tag: str
    chunk_id: str
    doc_id: str
    title: str
    section: str
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    retrieved: list[str]
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[S{i+1}] ({h.chunk.title} / {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def extractive_answer(question: str, hits: list[Hit], per_doc: int = 1) -> str:
    """Offline baseline: pick the best-overlapping sentence(s), keeping source tags.
    Favours sentences containing figures since fund questions are mostly quantitative."""
    q = set(tokenize(question))
    scored = []
    for i, h in enumerate(hits):
        name = set(tokenize(h.chunk.title))
        for s in _sentences(h.chunk.text):
            st = set(tokenize(s))
            # metric/subject terms count 3x more than fund-name terms
            score = 3 * len((q - name) & st) + 0.5 * len(q & name & st) + (0.5 if NUM.search(s) else 0) - 0.02 * len(st)
            scored.append((score, i, h.chunk.doc_id, s))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "Not found in the provided reports."
    picked, seen_docs = [], {}
    top = scored[0][0]
    for score, i, doc, s in scored:
        if score < top * 0.6 or seen_docs.get(doc, 0) >= per_doc:
            continue
        seen_docs[doc] = seen_docs.get(doc, 0) + 1
        picked.append(f"{s} [S{i+1}]")
        if len(picked) >= 2:
            break
    return " ".join(picked)


def resolve_citations(text: str, hits: list[Hit]) -> list[Citation]:
    out, seen = [], set()
    for m in CITE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= len(hits) and n not in seen:
            seen.add(n)
            c = hits[n - 1].chunk
            out.append(Citation(f"S{n}", c.chunk_id, c.doc_id, c.title, c.section, c.text[:280]))
    return out


class RAG:
    def __init__(self, retriever: Retriever, llm: LLM | None = None, k: int = 5):
        self.retriever, self.llm, self.k = retriever, llm or LLM(), k

    def retrieve(self, question: str, k: int) -> list[Hit]:
        """Hybrid search; if the question names funds, pin results to those funds."""
        funds = resolve_funds(question, self.retriever.chunks)
        if not funds:
            return self.retriever.search(question, k)
        per = max(2, k // len(funds))
        hits: list[Hit] = []
        for d in funds:
            hits += self.retriever.search(question, per, doc_filter=d)
        hits.sort(key=lambda h: h.rank)
        return [Hit(h.chunk, h.score, i + 1) for i, h in enumerate(hits[:max(k, per * len(funds))])]

    def answer(self, question: str, k: int | None = None, feedback: str = "", hits: list[Hit] | None = None) -> Answer:
        t0 = time.perf_counter()
        hits = hits or self.retrieve(question, k or self.k)
        usage = Usage()
        if self.llm.offline:
            per_doc = 1 if len(set(h.chunk.doc_id for h in hits[:3])) > 1 else 2
            text = extractive_answer(question, hits, per_doc=per_doc)
            prompt = SYSTEM + format_sources(hits) + question
            usage = Usage(estimate_tokens(prompt), estimate_tokens(text), 0.0, 0.0, 1)
        else:
            user = f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"
            if feedback:
                user += f"\n\nA reviewer flagged problems with a previous draft. Fix them:\n{feedback}"
            text, usage = self.llm.complete(SYSTEM, user, max_tokens=500)
        dt = time.perf_counter() - t0
        usage.latency_s = dt
        return Answer(question, text.strip(), resolve_citations(text, hits),
                      [h.chunk.chunk_id for h in hits], usage, dt)
