"""Build a QLoRA SFT dataset from *synthetic* fund reports with known ground truth.

Leakage control: the evaluation corpus (data/reports) is never used for training.
Phrasing is varied deliberately so the model learns the task, not the regexes.
"""
from __future__ import annotations

import json
import random
import re
import tempfile
from pathlib import Path

from .. import config
from ..extraction import EXTRACT_SYSTEM, METRIC_FIELDS
from ..ingest import ingest_file

NAMES = ["Larkspur", "Ironbridge", "Solent", "Kestrel", "Brookhaven", "Tamarind", "Oakmere", "Vantor",
         "Calder", "Westfold", "Pinecrest", "Rowan", "Galloway", "Hartwell", "Elmstead", "Quarry Lane"]
STRATS = ["Buyout", "Growth Equity", "Infrastructure", "Private Credit", "Secondaries", "Venture"]
CCY = [("USD", "$", ""), ("EUR", "EUR ", "")]


def _money(sym: str, m: float) -> str:
    return f"{sym}{m/1000:g} billion" if m >= 1000 else f"{sym}{m:.0f} million"


def synth_report(rng: random.Random, i: int) -> tuple[str, dict]:
    name = f"{rng.choice(NAMES)} {rng.choice(['Capital', 'Partners', 'Investors'])} Fund {rng.choice(['I','II','III','IV','V','VI'])}"
    strat, (ccy, sym, _) = rng.choice(STRATS), rng.choice(CCY)
    vintage = rng.randint(2015, 2024)
    commit = rng.choice([400, 650, 900, 1200, 1500, 2000, 2600, 3200])
    called = round(commit * rng.uniform(0.35, 0.95) / 10) * 10
    dpi, rvpi = round(rng.uniform(0.0, 1.2), 2), round(rng.uniform(0.4, 1.3), 2)
    tvpi = round(dpi + rvpi, 2)
    nav = round(called * rvpi / 10) * 10
    net = round(rng.uniform(-2, 25), 1)
    gross = round(net + rng.uniform(2, 7), 1) if rng.random() > 0.3 else None
    fee, carry = rng.choice([1.0, 1.25, 1.5, 1.75, 2.0]), rng.choice([10, 12.5, 15, 20, 25])
    gold = {"commitments_m": commit, "called_m": called, "nav_m": nav, "net_irr_pct": net, "gross_irr_pct": gross,
            "tvpi_x": tvpi, "dpi_x": dpi, "rvpi_x": rvpi, "mgmt_fee_pct": fee, "carry_pct": carry}
    perf = rng.choice([
        f"The fund's net IRR was {net}%" + (f" (gross {gross}%)" if gross else "") + f". TVPI came in at {tvpi}x, comprising DPI of {dpi}x and RVPI of {rvpi}x.",
        f"Since inception the fund has returned a net IRR of {net}%" + (f" and a gross IRR of {gross}%" if gross else "") + f". Net TVPI: {tvpi}x; DPI: {dpi}x; RVPI: {rvpi}x.",
        f"Performance to date: {net}% net IRR" + (f", {gross}% gross IRR" if gross else "") + f", {tvpi}x net multiple (TVPI), {dpi}x DPI, {rvpi}x RVPI.",
    ])
    navs = rng.choice([f"NAV at period end was {_money(sym, nav)}.", f"Reported net asset value: {_money(sym, nav)}.",
                       f"The fund's net asset value stood at {_money(sym, nav)}."])
    ov = rng.choice([
        f"{name} is a {vintage} vintage {strat.lower()} fund with {_money(sym, commit)} of commitments. Capital called to date totals {_money(sym, called)}.",
        f"Fund size is {_money(sym, commit)} ({vintage} vintage, {strat.lower()}). Investors have contributed {_money(sym, called)} so far.",
    ])
    fees = rng.choice([f"Management fee: {fee}% per annum. Carried interest: {carry}% subject to an 8% hurdle.",
                       f"The GP charges a {fee}% management fee and receives {carry}% carry."])
    report = f"""---
doc_id: synth-{i:04d}
fund: {name}
strategy: {strat}
vintage: {vintage}
report_date: {rng.choice(['2025-12-31','2026-03-31','2026-06-30'])}
currency: {ccy}
---
# {name}, Quarterly Report

## Overview
{ov}

## Performance
{perf} {navs}

## Portfolio
The portfolio contains {rng.randint(5, 40)} investments. {rng.choice(['Two exits closed in the period.', 'No realizations occurred this quarter.', 'One add-on acquisition was completed.'])}

## Terms
{fees}
"""
    return report, gold


def _fmt_target(gold: dict, chunks) -> dict:
    out = {}
    for f in METRIC_FIELDS:
        v = gold[f]
        unit = "m" if f.endswith("_m") else "x" if f.endswith("_x") else "%"
        if v is None:
            out[f] = {"value": None, "unit": unit, "evidence_chunk": None, "evidence_text": None}
            continue
        suffix = {"x": "x", "%": "%", "m": ""}[unit]
        probes = [f"{v}{suffix}", f"{v:g}{suffix}"]
        if unit == "m":
            probes = [f"{v:.0f} million", f"{v/1000:g} billion"]
        ev = next(((c.chunk_id, s) for c in chunks for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", c.text)
                   if any(p in s for p in probes)), (None, None))
        out[f] = {"value": v, "unit": unit, "evidence_chunk": ev[0], "evidence_text": ev[1]}
    return out


def build(n: int = 600, seed: int = 7, out: Path = config.ARTIFACTS / "sft") -> Path:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            text, gold = synth_report(rng, i)
            p = Path(td) / f"synth_{i}.md"
            p.write_text(text)
            chunks = ingest_file(p)
            meta = chunks[0].meta
            target = {"fund_name": meta["fund"], "strategy": meta["strategy"], "vintage": int(meta["vintage"]),
                      "as_of": meta["report_date"], "currency": meta["currency"], **_fmt_target(gold, chunks)}
            body = "\n\n".join(f"<chunk id='{c.chunk_id}' section='{c.section}'>\n{c.text}\n</chunk>" for c in chunks)
            rows.append({"messages": [{"role": "system", "content": EXTRACT_SYSTEM},
                                      {"role": "user", "content": body},
                                      {"role": "assistant", "content": json.dumps(target)}]})
    rng.shuffle(rows)
    cut = int(0.9 * len(rows))
    (out / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:cut]))
    (out / "val.jsonl").write_text("\n".join(json.dumps(r) for r in rows[cut:]))
    return out


if __name__ == "__main__":
    print(build())
