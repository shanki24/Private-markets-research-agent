import ask from "../netlify/functions/ask.mjs";
import ex from "../netlify/functions/extract.mjs";
const call = async (fn, body) => (await fn(new Request("http://x", { method: "POST", body: JSON.stringify(body) }))).json();
for (const [q, agent] of [["Which fund has the higher net TVPI, Northvale Fund III or Ashgrove Secondaries Fund V?", "planner"],
  ["What is the DPI of Halden Infrastructure Fund II?", "refine"], ["How many loans does Meridian hold?", "rag"],
  ["Compare the net IRR of Meridian Private Credit Fund I and Halden Infrastructure Fund II.", "planner"]]) {
  const r = await call(ask, { question: q, agent });
  console.log(agent, "|", r.answer, "|", r.citations.map((c) => c.chunk_id).join(","), "| score", r.evaluation.score.toFixed(2), r.error || "");
}
const e = await call(ex, { doc_id: "northvale-iii-q2-2026" });
console.log(e.net_irr_pct.value, e.tvpi_x.value, e.checks);
