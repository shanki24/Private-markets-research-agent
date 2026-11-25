// JS mirror of the Python core: BM25 retrieval, fund routing, extractive answers,
// rule extraction + checks, heuristic planner, deterministic evaluator.
import CHUNKS from "./index-data.mjs";

const STOP = new Set(`a an the of and or to in on for with by at as is are was were be been this that these those
it its from what which who how much many did does do has have had than fund funds`.split(/\s+/));
export const tokenize = (t) => (t.toLowerCase().match(/[a-z0-9]+(?:\.[0-9]+)?/g) || []).filter((w) => !STOP.has(w));

// ---------- BM25 ----------
const K1 = 1.4, B = 0.75;
const DOCS = CHUNKS.map((c) => tokenize(`${c.title} ${c.section} ${c.text}`));
const AVGDL = DOCS.reduce((s, d) => s + d.length, 0) / DOCS.length;
const DF = new Map();
DOCS.forEach((d) => new Set(d).forEach((t) => DF.set(t, (DF.get(t) || 0) + 1)));
const IDF = new Map([...DF].map(([t, f]) => [t, Math.log(1 + (DOCS.length - f + 0.5) / (f + 0.5))]));
const TF = DOCS.map((d) => d.reduce((m, t) => m.set(t, (m.get(t) || 0) + 1), new Map()));

export function search(query, k = 5, docFilter = null) {
  const q = tokenize(query);
  const scored = DOCS.map((d, i) => {
    const norm = K1 * (1 - B + (B * d.length) / AVGDL);
    let s = 0;
    for (const t of q) { const f = TF[i].get(t); if (f) s += IDF.get(t) * (f * (K1 + 1)) / (f + norm); }
    return { chunk: CHUNKS[i], score: s };
  }).filter((h) => !docFilter || h.chunk.doc_id === docFilter);
  return scored.sort((a, b) => b.score - a.score).slice(0, k);
}

// ---------- fund routing ----------
const ALIASES = (() => {
  const docTok = {};
  for (const c of CHUNKS) (docTok[c.doc_id] ||= new Set()); 
  for (const c of CHUNKS) tokenize(`${c.title} ${c.meta.manager || ""}`).forEach((t) => docTok[c.doc_id].add(t));
  const owners = {};
  for (const [d, ts] of Object.entries(docTok)) ts.forEach((t) => (owners[t] ||= new Set()).add(d));
  return Object.fromEntries(Object.entries(docTok).map(([d, ts]) =>
    [d, new Set([...ts].filter((t) => owners[t].size === 1 && !/^\d+$/.test(t) && t.length > 3))]));
})();
export const FUNDS = Object.fromEntries([...new Set(CHUNKS.map((c) => c.doc_id))]
  .map((d) => [d, CHUNKS.find((c) => c.doc_id === d).meta.fund]));

export function resolveFunds(q) {
  const found = [];
  for (const t of tokenize(q)) for (const [d, al] of Object.entries(ALIASES)) if (al.has(t) && !found.includes(d)) found.push(d);
  return found;
}

export function retrieve(question, k = 5) {
  const funds = resolveFunds(question);
  if (!funds.length) return search(question, k);
  const per = Math.max(2, Math.floor(k / funds.length));
  return funds.flatMap((d) => search(question, per, d)).sort((a, b) => b.score - a.score);
}

// ---------- extractive answer (offline baseline) ----------
export const NUM = /(?:EUR\s*|\$)?\d[\d,]*(?:\.\d+)?\s*(?:%|x|bps|basis points|MW|million|billion)?/gi;
const sentences = (t) => t.split(/(?<=[.!?])\s+(?=[A-Z])/).filter((s) => s.trim());

export function extractive(question, hits, perDoc = 1) {
  const q = new Set(tokenize(question));
  const scored = [];
  hits.forEach((h, i) => {
    const name = new Set(tokenize(h.chunk.title));
    for (const s of sentences(h.chunk.text)) {
      const st = new Set(tokenize(s));
      let sc = 0;
      for (const t of q) if (st.has(t)) sc += name.has(t) ? 0.5 : 3;
      sc += (s.match(NUM) ? 0.5 : 0) - 0.02 * st.size;
      scored.push({ sc, i, doc: h.chunk.doc_id, s });
    }
  });
  scored.sort((a, b) => b.sc - a.sc);
  if (!scored.length) return "Not found in the provided reports.";
  const top = scored[0].sc, seen = {}, out = [];
  for (const x of scored) {
    if (x.sc < top * 0.6 || (seen[x.doc] || 0) >= perDoc) continue;
    seen[x.doc] = (seen[x.doc] || 0) + 1;
    out.push(`${x.s} [S${x.i + 1}]`);
    if (out.length >= 2) break;
  }
  return out.join(" ");
}

export function resolveCitations(text, hits) {
  const seen = new Set(), out = [];
  for (const m of text.matchAll(/\[S(\d+)\]/g)) {
    const n = +m[1];
    if (n >= 1 && n <= hits.length && !seen.has(n)) {
      seen.add(n);
      const c = hits[n - 1].chunk;
      out.push({ tag: `S${n}`, chunk_id: c.chunk_id, doc_id: c.doc_id, title: c.title, section: c.section });
    }
  }
  return out;
}

// ---------- extraction ----------
export const METRIC_FIELDS = ["commitments_m", "called_m", "nav_m", "net_irr_pct", "gross_irr_pct",
  "tvpi_x", "dpi_x", "rvpi_x", "mgmt_fee_pct", "carry_pct"];
const MONEY = String.raw`(?:EUR\s*|\$)\s*([\d,.]+)\s*(million|billion)`;
const PATTERNS = {
  commitments_m: `commitments of ${MONEY}|total commitments of ${MONEY}`,
  called_m: `called(?: capital (?:was|of))?\\s*${MONEY}|has called ${MONEY}|had called ${MONEY}`,
  nav_m: `net asset value (?:was|stood at|of)\\s*${MONEY}`,
  net_irr_pct: String.raw`net IRR (?:of|was)\s*([\d.]+)%`,
  gross_irr_pct: String.raw`gross IRR (?:of|was)\s*([\d.]+)%`,
  tvpi_x: String.raw`TVPI (?:was|stood at|of)\s*([\d.]+)x`,
  dpi_x: String.raw`DPI (?:was|of)\s*([\d.]+)x`,
  rvpi_x: String.raw`RVPI (?:was|of)\s*([\d.]+)x`,
  mgmt_fee_pct: String.raw`management fee (?:is|stepped down from [\d.]+% of \w+ to)\s*([\d.]+)%`,
  carry_pct: String.raw`(?:Carried interest|Incentive fee) is\s*([\d.]+)%`,
};
export const docChunks = (docId) => CHUNKS.filter((c) => c.doc_id === docId);

export function extractRules(docId) {
  const cs = docChunks(docId), meta = cs[0].meta;
  const m = { doc_id: docId, fund_name: meta.fund, strategy: meta.strategy, vintage: +meta.vintage,
    as_of: meta.report_date, currency: meta.currency };
  for (const [f, pat] of Object.entries(PATTERNS)) {
    m[f] = { value: null };
    for (const c of cs) {
      const hit = c.text.match(new RegExp(pat, "i"));
      if (hit) {
        const g = hit.slice(1).filter(Boolean);
        let v = parseFloat(g[0].replace(/,/g, ""));
        if (g[1] && g[1].toLowerCase() === "billion") v *= 1000;
        m[f] = { value: v, unit: f.endsWith("_m") ? "m" : f.endsWith("_x") ? "x" : "%",
          evidence_chunk: c.chunk_id, evidence_text: hit[0] };
        break;
      }
    }
  }
  return validate(m);
}

export function validate(m) {
  const v = (f) => m[f]?.value, num = (x) => typeof x === "number";
  const checks = [];
  if (["tvpi_x", "dpi_x", "rvpi_x"].every((f) => num(v(f)))) {
    const gap = Math.abs(v("tvpi_x") - (v("dpi_x") + v("rvpi_x")));
    checks.push(`${gap <= 0.02 ? "PASS" : "FAIL"}: TVPI = DPI + RVPI (gap ${gap.toFixed(2)})`);
  }
  if (num(v("called_m")) && num(v("commitments_m")))
    checks.push(`${v("called_m") <= v("commitments_m") ? "PASS" : "FAIL"}: called capital <= commitments`);
  if (num(v("net_irr_pct")) && num(v("gross_irr_pct")))
    checks.push(`${v("net_irr_pct") <= v("gross_irr_pct") ? "PASS" : "FAIL"}: net IRR <= gross IRR`);
  const missing = METRIC_FIELDS.filter((f) => v(f) == null);
  if (missing.length) checks.push(`WARN: missing ${missing.join(", ")}`);
  m.checks = checks;
  return m;
}

// ---------- planner ----------
const METRIC_WORDS = { tvpi_x: ["tvpi", "total value"], dpi_x: ["dpi"], rvpi_x: ["rvpi"],
  net_irr_pct: ["net irr", "irr"], gross_irr_pct: ["gross irr"], nav_m: ["nav", "net asset value"],
  commitments_m: ["commitments", "fund size"], called_m: ["called"], mgmt_fee_pct: ["management fee"],
  carry_pct: ["carry", "carried interest"] };
export const LOWER_IS_BETTER = new Set(["mgmt_fee_pct", "carry_pct"]);
export const LABELS = { tvpi_x: "net TVPI", dpi_x: "DPI", rvpi_x: "RVPI", net_irr_pct: "net IRR",
  gross_irr_pct: "gross IRR", nav_m: "NAV", commitments_m: "commitments", called_m: "called capital",
  mgmt_fee_pct: "management fee", carry_pct: "carried interest / incentive fee" };

export function detectMetric(q) {
  const ql = q.toLowerCase();
  const entries = Object.entries(METRIC_WORDS).sort((a, b) =>
    Math.max(...b[1].map((w) => w.length)) - Math.max(...a[1].map((w) => w.length)));
  for (const [m, ws] of entries) if (ws.some((w) => ql.includes(w))) return m;
  return null;
}

export function heuristicPlan(question) {
  const funds = resolveFunds(question), metric = detectMetric(question);
  if (funds.length >= 2 && metric) {
    const steps = funds.map((f, i) => ({ id: `s${i + 1}`, tool: "extract_metric", fund: f, metric }));
    steps.push({ id: `s${steps.length + 1}`, tool: "compare", metric, inputs: steps.map((s) => s.id) });
    return steps;
  }
  if (funds.length >= 2) return funds.map((f, i) => ({ id: `s${i + 1}`, tool: "search_answer", query: question, fund: f }));
  return [{ id: "s1", tool: "search_answer", query: question, fund: funds[0] || null }];
}

// ---------- evaluator ----------
const normNum = (s) => s.toLowerCase().replace(/[,\s]/g, "");
export function evaluate(text, citations) {
  const sents = sentences(text).filter((s) => s.split(/\s+/).length > 2);
  const cov = sents.filter((s) => /\[S\d+\]/.test(s)).length / Math.max(sents.length, 1);
  const cited = citations.map((c) => CHUNKS.find((x) => x.chunk_id === c.chunk_id)?.text || "").join(" ");
  const nums = (text.replace(/\[S\d+\]/g, "").match(NUM) || []).map((n) => n.trim())
    .filter((n) => /\d/.test(n) && !/^(\d{1,2}|20\d\d)$/.test(n));
  const grounded = nums.filter((n) => normNum(cited).includes(normNum(n.replace(/[a-z%]+$/i, "")))).length;
  const ground = nums.length ? grounded / nums.length : 1;
  const answered = /not found/i.test(text) || !text.trim() ? 0 : 1;
  const issues = [];
  if (cov < 1) issues.push(`${Math.round((1 - cov) * sents.length)} sentence(s) lack citations`);
  if (ground < 1) issues.push(`${nums.length - grounded} figure(s) not found in cited sources`);
  if (!answered) issues.push("no answer produced");
  return { score: (cov + ground + answered) / 3, citation_coverage: cov, numeric_grounding: ground, answered, issues };
}

export { CHUNKS };
