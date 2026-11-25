"""Sparse (BM25), dense (sentence-transformers or TF-IDF/LSA fallback) and hybrid (RRF) retrieval."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .ingest import Chunk

STOP = set("""a an the of and or to in on for with by at as is are was were be been this that these those
it its from what which who how much many did does do has have had than fund funds""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [t for t in toks if t not in STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class BM25:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # Title + section are indexed with the text so fund names always match.
        self.docs = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for t in q:
                if t in tf:
                    s[i] += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
        return s


class Dense:
    """Uses sentence-transformers if installed, else TF-IDF + LSA (no downloads)."""

    def __init__(self, chunks: list[Chunk], model_name: str = "BAAI/bge-small-en-v1.5"):
        self.chunks = chunks
        texts = [f"{c.title}. {c.section}. {c.text}" for c in chunks]
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.name = f"dense:{model_name}"
            self.mat = self.model.encode(texts, normalize_embeddings=True)
            self._enc = lambda q: self.model.encode([q], normalize_embeddings=True)[0]
        except Exception:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            self.name = "dense:tfidf-lsa"
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            x = self.vec.fit_transform(texts)
            k = max(2, min(128, x.shape[0] - 1, x.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=k, random_state=0)
            self.mat = normalize(self.svd.fit_transform(x))
            self._enc = lambda q: normalize(self.svd.transform(self.vec.transform([q])))[0]

    def scores(self, query: str) -> np.ndarray:
        return self.mat @ self._enc(query)


class Retriever:
    def __init__(self, chunks: list[Chunk], kind: str = "hybrid", rrf_k: int = 60):
        self.chunks, self.kind, self.rrf_k = chunks, kind, rrf_k
        self.bm25 = BM25(chunks) if kind in ("bm25", "hybrid") else None
        self.dense = Dense(chunks) if kind in ("dense", "hybrid") else None

    @property
    def name(self) -> str:
        if self.kind == "hybrid":
            return f"hybrid(bm25+{self.dense.name},rrf)"
        return self.bm25.name if self.bm25 else self.dense.name

    def search(self, query: str, k: int = 5, doc_filter: str | None = None) -> list[Hit]:
        if self.kind == "bm25":
            s = self.bm25.scores(query)
        elif self.kind == "dense":
            s = self.dense.scores(query)
        else:  # reciprocal rank fusion
            s = np.zeros(len(self.chunks))
            for comp in (self.bm25.scores(query), self.dense.scores(query)):
                for r, i in enumerate(np.argsort(-comp)):
                    s[i] += 1.0 / (self.rrf_k + r + 1)
        order = [i for i in np.argsort(-s) if not doc_filter or self.chunks[i].doc_id == doc_filter]
        return [Hit(self.chunks[i], float(s[i]), r + 1) for r, i in enumerate(order[:k])]


def fund_aliases(chunks: list[Chunk]) -> dict[str, set[str]]:
    """Tokens that identify exactly one document (e.g. 'northvale', 'halden')."""
    doc_tokens: dict[str, set[str]] = {}
    for c in chunks:
        doc_tokens.setdefault(c.doc_id, set()).update(tokenize(c.title + " " + c.meta.get("manager", "")))
    owners: dict[str, set[str]] = {}
    for d, toks in doc_tokens.items():
        for t in toks:
            owners.setdefault(t, set()).add(d)
    return {d: {t for t in toks if len(owners[t]) == 1 and not t.isdigit() and len(t) > 3}
            for d, toks in doc_tokens.items()}


def resolve_funds(question: str, chunks: list[Chunk]) -> list[str]:
    q = tokenize(question)
    aliases = fund_aliases(chunks)
    found = []
    for t in q:  # keep mention order
        for d, al in aliases.items():
            if t in al and d not in found:
                found.append(d)
    return found
