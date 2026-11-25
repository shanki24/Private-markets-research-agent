# Private Markets Research Agent

Research assistant for private-market fund reports: ingestion, hybrid retrieval with source
citations, two agent patterns, structured extraction behind human approval, a benchmark
harness, a QLoRA fine-tuning experiment and a model registry. The `web/` site plus
`netlify/functions/` deploy to Netlify as a live demo.

```
data/reports/*.md|pdf ──► ingest (section-aware chunks + metadata)
                              │
                  ┌───────────┴───────────┐
             BM25 (sparse)          Dense (bge / TF-IDF-LSA)
                  └──── RRF fusion + fund routing ────┘
                              │
        ┌─────────────────────┼─────────────────────────┐
   RAG (single pass)   Planner–executor            Evaluator–optimizer
   cited answer        plan → tools → synthesize   draft → score → fix → redraft
                       (compare computed in code)  (coverage, grounding, LLM judge)
                              │
     Extraction (pydantic schema, per-field evidence, consistency checks)
                              │
     Human approval queue (SQLite audit trail; flagged records first)
                              │
     Benchmarks ─► Registry (model versions × runs × dataset hash, optional MLflow)
                              ▲
     QLoRA experiment (synthetic training data, held-out real reports)
```

## Layout

| Path | What it does |
|---|---|
| `backend/pmra/ingest.py` | Markdown/PDF loading, front-matter metadata, sentence-aligned overlapping chunks |
| `backend/pmra/retriever.py` | BM25, dense, hybrid RRF, fund-name resolver |
| `backend/pmra/rag.py` | Cited answering; offline extractive baseline |
| `backend/pmra/llm.py` | Anthropic / OpenAI / local HF (+LoRA adapter) clients with token, cost, latency tracking |
| `backend/pmra/agents/planner_executor.py` | Typed plans: `search_answer`, `extract_metric`, `compare` |
| `backend/pmra/agents/evaluator_optimizer.py` | Scored refinement loop with query rewrite and wider retrieval |
| `backend/pmra/extraction.py` | `FundMetrics` schema, LLM + rules extractors, TVPI = DPI + RVPI and other checks |
| `backend/pmra/approval.py` | Approve / reject / correct records, with reviewer, notes and timestamps |
| `backend/pmra/eval/benchmark.py` | Recall@k, MRR, nDCG, hit@1, accuracy, citation precision/recall, p50/p95 latency, tokens, cost |
| `backend/pmra/registry.py` | Model versions and evaluation runs |
| `backend/pmra/finetune/` | Synthetic SFT data, QLoRA trainer, base-vs-adapter evaluation |
| `netlify/functions/` | `/api/ask` and `/api/extract`, a JS mirror of the core |
| `web/` | Static site: ask, review, benchmarks, registry |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd backend
python -m pmra.cli ingest
python -m pmra.cli ask "Rank Northvale, Halden and Corvid by net IRR" --agent planner
python -m pmra.cli bench            # writes artifacts/bench_*.json and registry rows
python -m pmra.cli extract --submit # queue extractions for approval
python -m pmra.cli review --reviewer you
python -m pmra.cli export           # refresh web/data and the functions' index
python -m pytest ../tests -q
cd .. && npm run local              # http://localhost:8888, no install needed
```

Use Claude instead of the offline baseline:

```bash
export PMRA_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
export PMRA_MODEL=claude-sonnet-5-5       # any model id your key can use
python -m pmra.cli bench                  # logged as a separate model version
python -m pmra.cli export
```

Set real per-token prices (`PRICE_SONNET_IN`, `PRICE_SONNET_OUT`, etc.) before reporting
cost; the defaults in `config.py` are placeholders.

## Deploy to Netlify

1. Push the repo to GitHub (the generated `web/data/*.json` and
   `netlify/functions/lib/index-data.mjs` are committed, so Netlify needs no Python).
2. Netlify → Add new site → Import from Git. `netlify.toml` sets publish dir `web` and
   functions dir `netlify/functions`; leave build settings as detected.
3. Optional, for LLM mode: Site configuration → Environment variables →
   `ANTHROPIC_API_KEY`, `PMRA_MODEL`, `PRICE_IN_PER_MTOK`, `PRICE_OUT_PER_MTOK`.
   Without a key the site runs the offline baseline, so the demo never breaks.
4. Or from the CLI: `npm i -g netlify-cli && netlify deploy --prod`.

Function calls to Claude can take several seconds. Netlify's default synchronous function
timeout is short, so if planner + synthesis runs time out, raise the function timeout in
your Netlify plan settings or use the "Single pass" agent.

## Adding your own reports

Drop `.md`, `.txt` or `.pdf` files into `data/reports/`. Markdown can carry front matter
(`doc_id`, `fund`, `manager`, `strategy`, `vintage`, `report_date`, `currency`). Then run
`ingest`, `export`, and add gold questions to `data/eval/qa.jsonl` and values to
`data/eval/extraction_gold.json` to benchmark them.

## QLoRA experiment

Training needs a CUDA GPU (a free Colab T4 handles Qwen2.5-0.5B):

```bash
pip install -r requirements-finetune.txt
cd backend
python -m pmra.finetune.build_dataset                     # 600 synthetic reports → 540 train / 60 val
python -m pmra.finetune.qlora_train --base Qwen/Qwen2.5-0.5B-Instruct --epochs 2
python -m pmra.finetune.eval_ft --base Qwen/Qwen2.5-0.5B-Instruct --adapter artifacts/adapters/<run>
python -m pmra.cli export                                 # comparison appears on the Models page
```

Design choices: 4-bit NF4 with double quantization, LoRA r=16/α=32 on all attention and
MLP projections, loss masked to the assistant JSON only, cosine schedule, paged 8-bit
AdamW. Training data is generated from templates with varied phrasing and known ground
truth; the five real-format reports are never used for training, so the comparison is a
genuine held-out test. Hyperparameters, train-data hash, GPU and wall time are recorded
in the registry with the adapter.

## Reading the current benchmark honestly

The committed results are from the offline baseline on 15 test questions and 5 reports:

- BM25 beats the hybrid here because the dense side is TF-IDF/LSA (no embedding model
  downloaded). Install `sentence-transformers` and rerun to measure real hybrid retrieval.
- Single-pass RAG misses comparison questions (50%); the planner fixes them (100%)
  by extracting each figure and ranking in code.
- Extraction scores 100% because the rule extractor was written against these report
  formats. Treat it as a regression test, not a generalization claim; the QLoRA and LLM
  runs are the meaningful extraction comparison.
- The corpus is synthetic and small. Expand `qa.jsonl` before drawing conclusions.
