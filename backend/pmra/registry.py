"""Model version + evaluation run registry (SQLite; mirrors to MLflow when installed)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone

from . import config

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS models(version_id TEXT PRIMARY KEY, kind TEXT, base TEXT,
       params TEXT, artifact_uri TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, version_id TEXT, suite TEXT,
       dataset_hash TEXT, config TEXT, metrics TEXT, git_sha TEXT, created_at TEXT)""",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def file_hash(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


class Registry:
    def __init__(self, path=config.REGISTRY_DB):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        for s in SCHEMA:
            self.db.execute(s)

    def register_model(self, version_id: str, kind: str, base: str = "", params: dict | None = None,
                       artifact_uri: str = ""):
        self.db.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?,?)",
                        (version_id, kind, base, json.dumps(params or {}), artifact_uri,
                         datetime.now(timezone.utc).isoformat()))
        self.db.commit()

    def log_run(self, version_id: str, suite: str, metrics: dict, cfg: dict, dataset_hash: str) -> str:
        rid = uuid.uuid4().hex[:10]
        self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                        (rid, version_id, suite, dataset_hash, json.dumps(cfg), json.dumps(metrics),
                         _git_sha(), datetime.now(timezone.utc).isoformat()))
        self.db.commit()
        try:  # optional MLflow mirror
            import mlflow
            with mlflow.start_run(run_name=f"{suite}:{version_id}"):
                mlflow.log_params({**cfg, "version_id": version_id, "dataset_hash": dataset_hash})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception:
            pass
        return rid

    def export(self) -> dict:
        models = [dict(r) | {"params": json.loads(r["params"])} for r in self.db.execute("SELECT * FROM models")]
        runs = [dict(r) | {"metrics": json.loads(r["metrics"]), "config": json.loads(r["config"])}
                for r in self.db.execute("SELECT * FROM runs ORDER BY created_at")]
        return {"models": models, "runs": runs}
