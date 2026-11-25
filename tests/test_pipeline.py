"""Run: cd backend && python -m pytest ../tests -q"""
from pmra.agents.evaluator_optimizer import EvaluatorOptimizer
from pmra.agents.planner_executor import PlannerExecutor
from pmra.approval import ApprovalQueue
from pmra.extraction import extract
from pmra.ingest import ingest_dir
from pmra.llm import LLM
from pmra.rag import RAG
from pmra.retriever import Retriever, resolve_funds

CHUNKS = ingest_dir()
RAG_ = RAG(Retriever(CHUNKS, "hybrid"), LLM("offline"))


def test_fund_routing():
    assert resolve_funds("Compare Northvale and Ashgrove", CHUNKS) == ["northvale-iii-q2-2026", "ashgrove-sec-v-q2-2026"]


def test_rag_cites_correct_passage():
    a = RAG_.answer("What is the DPI of Halden Infrastructure Fund II?")
    assert "0.18x" in a.text and a.citations[0].doc_id == "halden-infra-ii-fy2025"


def test_planner_computes_comparison():
    a = PlannerExecutor(RAG_).run("Which has the higher net TVPI, Northvale or Ashgrove?")
    assert a.text.startswith("Northvale") and {c.doc_id for c in a.citations} == {"northvale-iii-q2-2026", "ashgrove-sec-v-q2-2026"}


def test_evaluator_scores_grounded_answer():
    a = EvaluatorOptimizer(RAG_).run("What gross MOIC did Northvale realize on Ridgeline Logistics?")
    assert "3.1x" in a.text and a.trace[-1]["score"] >= 0.9


def test_extraction_and_checks():
    m, _ = extract([c for c in CHUNKS if c.doc_id == "meridian-credit-i-q2-2026"], LLM("offline"))
    assert (m.tvpi_x.value, m.dpi_x.value, m.rvpi_x.value) == (1.31, 0.74, 0.57)
    assert any(c.startswith("PASS: TVPI") for c in m.checks)


def test_flagged_records_need_review(tmp_path):
    q = ApprovalQueue(tmp_path / "a.sqlite")
    m, _ = extract([c for c in CHUNKS if c.doc_id == "northvale-iii-q2-2026"], LLM("offline"))
    m.tvpi_x.value = 2.5  # inconsistent with DPI + RVPI
    from pmra.extraction import validate
    rid = q.submit(validate(m), "test")
    assert q.pending()[0]["status"] == "pending_flagged"
    q.decide(rid, True, "tester", edits={"tvpi_x": 1.86})
    assert q.approved()[0]["tvpi_x"]["value"] == 1.86
