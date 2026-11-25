// POST /api/ask  {question, agent: "rag"|"planner"|"refine", mode: "auto"|"offline"}
import { CHUNKS, FUNDS, LABELS, LOWER_IS_BETTER, METRIC_FIELDS, evaluate, extractRules, extractive,
  heuristicPlan, resolveCitations, retrieve, search } from "./lib/core.mjs";
import { addUsage, complete, estTokens, hasKey, MODEL, newUsage, parseJSON } from "./lib/llm.mjs";

const SYSTEM = `You are a private-markets research analyst. Answer ONLY from the numbered sources.
Rules:
- Cite every factual sentence with source tags like [S1] or [S2][S3].
- Quote figures exactly as written in the sources (units, x, %, currency).
- If the sources do not contain the answer, say "Not found in the provided reports."
- Be concise: 1-4 sentences.`;
const fmtSources = (hits) => hits.map((h, i) => `[S${i + 1}] (${h.chunk.title} / ${h.chunk.section})\n${h.chunk.text}`).join("\n\n");

async function ragAnswer(question, { online, k = 5, feedback = "", hits = null }) {
  hits = hits || retrieve(question, k);
  let text, usage;
  if (!online) {
    const per = new Set(hits.slice(0, 3).map((h) => h.chunk.doc_id)).size > 1 ? 1 : 2;
    text = extractive(question, hits, per);
    usage = { input_tokens: estTokens(SYSTEM + fmtSources(hits) + question), output_tokens: estTokens(text), cost_usd: 0, calls: 1 };
  } else {
    let user = `Sources:\n${fmtSources(hits)}\n\nQuestion: ${question}`;
    if (feedback) user += `\n\nA reviewer flagged problems with a previous draft. Fix them:\n${feedback}`;
    ({ text, usage } = await complete(SYSTEM, user, 500));
  }
  return { text: text.trim(), hits, usage };
}

async function plannerRun(question, online) {
  const usage = newUsage(), sources = [], trace = [];
  const cite = (chunk) => {
    let i = sources.findIndex((h) => h.chunk.chunk_id === chunk.chunk_id);
    if (i < 0) { sources.push({ chunk }); i = sources.length - 1; }
    return `S${i + 1}`;
  };
  let steps = heuristicPlan(question), planner = "heuristic";
  if (online) {
    try {
      const sys = `You plan research over private-market fund reports. Return ONLY JSON:
{"steps":[{"id":"s1","tool":"search_answer","query":str,"fund":doc_id|null} | {"id":"s2","tool":"extract_metric","fund":doc_id,"metric":METRIC} | {"id":"s3","tool":"compare","inputs":[ids],"metric":METRIC}]}
Use extract_metric + compare for numeric comparisons or rankings across funds. Max 8 steps.`;
      const r = await complete(sys, `${JSON.stringify({ FUNDS, METRICS: METRIC_FIELDS })}\n\nQuestion: ${question}`, 600);
      addUsage(usage, r.usage);
      const p = parseJSON(r.text).steps;
      if (Array.isArray(p) && p.length) { steps = p; planner = MODEL; }
    } catch { /* fall back to heuristic plan */ }
  }
  const byId = {};
  for (const s of steps) {
    const t0 = Date.now();
    if (s.tool === "search_answer") {
      const q = s.query || question;
      const hits = s.fund ? search(q, 3, s.fund) : retrieve(q, 4);
      const a = await ragAnswer(q, { online, hits });
      addUsage(usage, a.usage);
      s.output = { answer: a.text.replace(/\[S(\d+)\]/g, (_, n) => (hits[+n - 1] ? `[${cite(hits[+n - 1].chunk)}]` : "")) };
    } else if (s.tool === "extract_metric") {
      const m = extractRules(s.fund), f = m[s.metric] || { value: null };
      const ch = CHUNKS.find((c) => c.chunk_id === f.evidence_chunk);
      s.output = { fund: FUNDS[s.fund], value: f.value, unit: f.unit, tag: ch ? cite(ch) : null };
    } else if (s.tool === "compare") {
      const rows = s.inputs.map((i) => byId[i]?.output).filter((o) => o && o.value != null);
      rows.sort((a, b) => (LOWER_IS_BETTER.has(s.metric) ? a.value - b.value : b.value - a.value));
      s.output = { ranking: rows, metric: s.metric };
    }
    s.latency_ms = Date.now() - t0;
    byId[s.id] = s;
    trace.push(s);
  }
  const fmt = (r) => `${r.unit === "x" ? r.value + "x" : r.unit === "%" ? r.value + "%" : r.value.toLocaleString() + "m"}${r.tag ? ` [${r.tag}]` : ""}`;
  const cmp = steps.find((s) => s.tool === "compare");
  let text;
  if (cmp && cmp.output.ranking.length) {
    const [lead, ...rest] = cmp.output.ranking;
    text = `${lead.fund} has the ${LOWER_IS_BETTER.has(cmp.metric) ? "lowest" : "highest"} ${LABELS[cmp.metric] || cmp.metric} at ${fmt(lead)}` +
      (rest.length ? `, compared with ${rest.map((r) => `${r.fund} ${fmt(r)}`).join("; ")}.` : ".");
  } else if (!online || steps.length === 1) {
    text = steps.filter((s) => s.tool === "search_answer").map((s) => s.output.answer).join(" ");
  } else {
    const notes = steps.map((s) => `- ${s.id} ${s.tool}: ${JSON.stringify(s.output)}`).join("\n");
    const r = await complete("Write a concise cited answer from the step results and sources. Cite with [S#]. Use only given facts.",
      `Question: ${question}\n\nStep results:\n${notes}\n\nSources:\n${fmtSources(sources)}`, 500);
    addUsage(usage, r.usage);
    text = r.text;
  }
  return { text, hits: sources, usage, trace: { planner, steps: trace } };
}

async function refineRun(question, online, threshold = 0.9, maxIters = 3) {
  const usage = newUsage(), iterations = [];
  let best = null, query = question, k = 5, feedback = "";
  for (let it = 1; it <= maxIters; it++) {
    const a = await ragAnswer(query, { online, k, feedback });
    addUsage(usage, a.usage);
    const ev = evaluate(a.text, resolveCitations(a.text, a.hits));
    if (online) {
      try {
        const cited = resolveCitations(a.text, a.hits).map((c) => CHUNKS.find((x) => x.chunk_id === c.chunk_id).text).join("\n");
        const j = await complete(`You grade an analyst answer against its sources. Return ONLY JSON: {"faithfulness":0-1,"completeness":0-1,"issues":[str]}`,
          `Question: ${question}\nAnswer: ${a.text}\n\nCited sources:\n${cited}`, 300);
        addUsage(usage, j.usage);
        const g = parseJSON(j.text);
        ev.judge = g;
        ev.issues.push(...(g.issues || []));
        ev.score = (ev.citation_coverage + ev.numeric_grounding + ev.answered + g.faithfulness + g.completeness) / 5;
      } catch { /* judge optional */ }
    }
    iterations.push({ iteration: it, query, k, draft: a.text, ...ev });
    if (!best || ev.score > best.score) best = { ...a, score: ev.score };
    if (ev.score >= threshold) break;
    if (!ev.answered || ev.numeric_grounding < 1) { query = `${question} net IRR TVPI DPI net asset value`; k += 3; }
    feedback = ev.issues.join("\n");
  }
  return { text: best.text, hits: best.hits, usage, trace: { iterations } };
}

export default async (req) => {
  if (req.method !== "POST") return new Response("POST only", { status: 405 });
  const t0 = Date.now();
  try {
    const { question, agent = "planner", mode = "auto" } = await req.json();
    if (!question || question.length > 600) return Response.json({ error: "Ask a question under 600 characters." }, { status: 400 });
    const online = mode !== "offline" && hasKey();
    const r = agent === "rag" ? await ragAnswer(question, { online })
      : agent === "refine" ? await refineRun(question, online) : await plannerRun(question, online);
    const citations = resolveCitations(r.text, r.hits);
    return Response.json({
      question, agent, model: online ? MODEL : "offline-extractive-v1", answer: r.text, citations,
      sources: r.hits.map((h, i) => ({ tag: `S${i + 1}`, chunk_id: h.chunk.chunk_id, doc_id: h.chunk.doc_id,
        title: h.chunk.title, section: h.chunk.section, text: h.chunk.text })),
      evaluation: evaluate(r.text, citations), usage: r.usage, trace: r.trace || null, latency_ms: Date.now() - t0,
    });
  } catch (e) {
    return Response.json({ error: String(e.message || e) }, { status: 500 });
  }
};

export const config = { path: "/api/ask" };
