"""Central configuration. Everything is overridable with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("PMRA_ROOT", Path(__file__).resolve().parents[2]))
REPORTS_DIR = ROOT / "data" / "reports"
EVAL_FILE = ROOT / "data" / "eval" / "qa.jsonl"
ARTIFACTS = ROOT / "artifacts"
WEB_DATA = ROOT / "web" / "data"
CHUNKS_FILE = ARTIFACTS / "chunks.jsonl"
REGISTRY_DB = ARTIFACTS / "registry.sqlite"
APPROVAL_DB = ARTIFACTS / "approvals.sqlite"

CHUNK_WORDS = int(os.getenv("PMRA_CHUNK_WORDS", 90))
CHUNK_OVERLAP = int(os.getenv("PMRA_CHUNK_OVERLAP", 20))
TOP_K = int(os.getenv("PMRA_TOP_K", 5))

# Provider: "offline" (deterministic extractive baseline, no key needed),
# "anthropic", "openai", or "hf:<path-or-hub-id>[@<adapter-path>]" for local models.
PROVIDER = os.getenv("PMRA_PROVIDER", "offline")
MODEL = os.getenv("PMRA_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.getenv("PMRA_JUDGE_MODEL", "claude-haiku-5-5")

# USD per million tokens (input, output). These are PLACEHOLDERS:
# check your provider's pricing page and set real values before reporting costs.
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (float(os.getenv("PRICE_SONNET_IN", 3.0)), float(os.getenv("PRICE_SONNET_OUT", 15.0))),
    "claude-haiku-5-5": (float(os.getenv("PRICE_HAIKU_IN", 1.0)), float(os.getenv("PRICE_HAIKU_OUT", 5.0))),
    "offline": (0.0, 0.0),
}

ARTIFACTS.mkdir(parents=True, exist_ok=True)
