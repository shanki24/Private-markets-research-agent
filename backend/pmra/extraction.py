"""Structured extraction of fund metrics with per-field evidence and consistency checks."""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field

from .ingest import Chunk
from .llm import LLM, Usage, estimate_tokens


class Field_(BaseModel):
    value: Optional[float | str] = None
    unit: Optional[str] = None
    evidence_chunk: Optional[str] = None
    evidence_text: Optional[str] = None


class FundMetrics(BaseModel):
    doc_id: str
    fund_name: Optional[str] = None
    strategy: Optional[str] = None
    vintage: Optional[int] = None
    as_of: Optional[str] = None
    currency: Optional[str] = None
    commitments_m: Field_ = Field(default_factory=Field_)
    called_m: Field_ = Field(default_factory=Field_)
    nav_m: Field_ = Field(default_factory=Field_)
    net_irr_pct: Field_ = Field(default_factory=Field_)
    gross_irr_pct: Field_ = Field(default_factory=Field_)
    tvpi_x: Field_ = Field(default_factory=Field_)
    dpi_x: Field_ = Field(default_factory=Field_)
    rvpi_x: Field_ = Field(default_factory=Field_)
    mgmt_fee_pct: Field_ = Field(default_factory=Field_)
    carry_pct: Field_ = Field(default_factory=Field_)
    checks: list[str] = Field(default_factory=list)


METRIC_FIELDS = ["commitments_m", "called_m", "nav_m", "net_irr_pct", "gross_irr_pct",
                 "tvpi_x", "dpi_x", "rvpi_x", "mgmt_fee_pct", "carry_pct"]

MONEY = r"(?:EUR\s*|\$)\s*([\d,.]+)\s*(million|billion)"
PATTERNS = {
    "commitments_m": rf"commitments of {MONEY}|total commitments of {MONEY}",
    "called_m": rf"called(?: capital (?:was|of))?\s*{MONEY}|has called {MONEY}|had called {MONEY}",
    "nav_m": rf"net asset value (?:was|stood at|of)\s*{MONEY}",
    "net_irr_pct": r"net IRR (?:of|was)\s*([\d.]+)%",
    "gross_irr_pct": r"gross IRR (?:of|was)\s*([\d.]+)%",
    "tvpi_x": r"TVPI (?:was|stood at|of)\s*([\d.]+)x",
    "dpi_x": r"DPI (?:was|of)\s*([\d.]+)x",
    "rvpi_x": r"RVPI (?:was|of)\s*([\d.]+)x",
    "mgmt_fee_pct": r"management fee (?:is|stepped down from [\d.]+% of \w+ to)\s*([\d.]+)%",
    "carry_pct": r"(?:Carried interest|Incentive fee) is\s*([\d.]+)%",
}


def _num(groups: tuple) -> float | None:
    g = [x for x in groups if x]
    if not g:
        return None
    v = float(g[0].replace(",", ""))
    if len(g) > 1 and g[1].lower() == "billion":
        v *= 1000
    return v


def extract_rules(chunks: list[Chunk]) -> FundMetrics:
    meta = chunks[0].meta
    m = FundMetrics(doc_id=chunks[0].doc_id, fund_name=meta.get("fund", chunks[0].title),
                    strategy=meta.get("strategy"), vintage=int(meta["vintage"]) if meta.get("vintage") else None,
                    as_of=meta.get("report_date"), currency=meta.get("currency"))
    for f, pat in PATTERNS.items():
        for c in chunks:
            hit = re.search(pat, c.text, re.I)
            if hit:
                unit = "m" if f.endswith("_m") else ("x" if f.endswith("_x") else "%")
                setattr(m, f, Field_(value=_num(hit.groups()), unit=unit, evidence_chunk=c.chunk_id,
                                     evidence_text=hit.group(0)))
                break
    return m


EXTRACT_SYSTEM = """Extract fund metrics from the report. Return ONLY JSON matching:
{"fund_name": str, "strategy": str, "vintage": int, "as_of": "YYYY-MM-DD", "currency": str,
 "<field>": {"value": number|null, "unit": str, "evidence_chunk": "<chunk id>", "evidence_text": "<exact short quote>"}}
for fields: """ + ", ".join(METRIC_FIELDS) + """.
Money fields are in millions of the fund currency. Percent fields as numbers (17.4 not 0.174).
Use null when a value is absent. Never guess."""


def extract_llm(chunks: list[Chunk], llm: LLM) -> tuple[FundMetrics, Usage]:
    body = "\n\n".join(f"<chunk id='{c.chunk_id}' section='{c.section}'>\n{c.text}\n</chunk>" for c in chunks)
    text, usage = llm.complete(EXTRACT_SYSTEM, body, max_tokens=1500)
    raw = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))
    raw["doc_id"] = chunks[0].doc_id
    return FundMetrics.model_validate(raw), usage


def validate(m: FundMetrics) -> FundMetrics:
    """Domain consistency checks; failures go to the human reviewer."""
    v = lambda f: getattr(m, f).value
    checks = []
    if all(isinstance(v(f), (int, float)) for f in ("tvpi_x", "dpi_x", "rvpi_x")):
        gap = abs(v("tvpi_x") - (v("dpi_x") + v("rvpi_x")))
        checks.append(("PASS" if gap <= 0.02 else "FAIL") + f": TVPI = DPI + RVPI (gap {gap:.2f})")
    if isinstance(v("called_m"), (int, float)) and isinstance(v("commitments_m"), (int, float)):
        ok = v("called_m") <= v("commitments_m")
        checks.append(("PASS" if ok else "FAIL") + ": called capital <= commitments")
    if isinstance(v("net_irr_pct"), (int, float)) and isinstance(v("gross_irr_pct"), (int, float)):
        ok = v("net_irr_pct") <= v("gross_irr_pct")
        checks.append(("PASS" if ok else "FAIL") + ": net IRR <= gross IRR")
    missing = [f for f in METRIC_FIELDS if v(f) is None]
    if missing:
        checks.append("WARN: missing " + ", ".join(missing))
    m.checks = checks
    return m


def extract(chunks: list[Chunk], llm: LLM | None = None) -> tuple[FundMetrics, Usage]:
    llm = llm or LLM()
    if llm.offline:
        m = extract_rules(chunks)
        u = Usage(sum(estimate_tokens(c.text) for c in chunks), 200, 0.0, 0.0, 1)
    else:
        m, u = extract_llm(chunks, llm)
    return validate(m), u
