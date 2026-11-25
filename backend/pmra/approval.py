"""Human-in-the-loop approval queue for extracted records (SQLite-backed, auditable)."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from . import config
from .extraction import FundMetrics

SCHEMA = """CREATE TABLE IF NOT EXISTS reviews(
  id TEXT PRIMARY KEY, doc_id TEXT, payload TEXT, status TEXT, model_version TEXT,
  reviewer TEXT, notes TEXT, edits TEXT, created_at TEXT, decided_at TEXT)"""


class ApprovalQueue:
    def __init__(self, path=config.APPROVAL_DB):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(SCHEMA)

    def submit(self, m: FundMetrics, model_version: str) -> str:
        rid = uuid.uuid4().hex[:10]
        # records with failed checks are flagged for closer review, never auto-approved
        status = "pending_flagged" if any(c.startswith("FAIL") for c in m.checks) else "pending"
        self.db.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (rid, m.doc_id, m.model_dump_json(), status, model_version, None, None, None,
                         datetime.now(timezone.utc).isoformat(), None))
        self.db.commit()
        return rid

    def pending(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM reviews WHERE status LIKE 'pending%' ORDER BY created_at").fetchall()

    def decide(self, rid: str, approve: bool, reviewer: str, notes: str = "", edits: dict | None = None):
        row = self.db.execute("SELECT payload FROM reviews WHERE id=?", (rid,)).fetchone()
        payload = json.loads(row["payload"])
        for field, value in (edits or {}).items():  # reviewer corrections are kept separately for audit
            payload.setdefault(field, {})["value"] = value
        self.db.execute("UPDATE reviews SET status=?, reviewer=?, notes=?, edits=?, payload=?, decided_at=? WHERE id=?",
                        ("approved" if approve else "rejected", reviewer, notes, json.dumps(edits or {}),
                         json.dumps(payload), datetime.now(timezone.utc).isoformat(), rid))
        self.db.commit()

    def approved(self) -> list[dict]:
        return [json.loads(r["payload"]) for r in
                self.db.execute("SELECT payload FROM reviews WHERE status='approved'").fetchall()]

    def review_cli(self, reviewer: str):
        for r in self.pending():
            m = json.loads(r["payload"])
            print(f"\n=== {r['id']}  {m['fund_name']}  [{r['status']}]  model={r['model_version']}")
            for k, v in m.items():
                if isinstance(v, dict) and "value" in v:
                    print(f"  {k:<15} {v['value']!s:<10} <- {v.get('evidence_text') or '-'}")
            for c in m["checks"]:
                print("  check:", c)
            choice = input("Approve [a], reject [r], edit [e], skip [s]? ").strip().lower()
            if choice == "e":
                edits = json.loads(input('Edits as JSON, e.g. {"nav_m": 960}: ') or "{}")
                self.decide(r["id"], True, reviewer, input("Notes: "), edits)
            elif choice in ("a", "r"):
                self.decide(r["id"], choice == "a", reviewer, input("Notes: "))
