"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x * 100).toFixed(1) + "%";
const fmt3 = (x) => (typeof x === "number" ? x.toFixed(3) : esc(x));
const toast = (msg) => { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2400); };

const state = { index: [], extractions: [], benchmarks: {}, registry: { models: [], runs: [] }, reviews: loadReviews() };
const FIELD_LABELS = { commitments_m: "Commitments (m)", called_m: "Called capital (m)", nav_m: "NAV (m)",
  net_irr_pct: "Net IRR (%)", gross_irr_pct: "Gross IRR (%)", tvpi_x: "Net TVPI (x)", dpi_x: "DPI (x)",
  rvpi_x: "RVPI (x)", mgmt_fee_pct: "Management fee (%)", carry_pct: "Carry / incentive (%)" };
const EXAMPLES = [
  "Rank Northvale, Halden, Corvid, Meridian and Ashgrove by DPI",
  "Which fund has the higher net TVPI, Northvale Fund III or Ashgrove Secondaries Fund V?",
  "What share of Meridian's loans are on non-accrual status?",
  "What discount to NAV did Ashgrove pay for the buyout fund interests?",
  "Which has the lowest management fee: Northvale, Halden or Corvid?",
];

async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }

// ---------------- routing ----------------
function route() {
  const v = (location.hash || "#ask").slice(1);
  const view = ["ask", "review", "benchmarks", "models"].includes(v) ? v : "ask";
  document.querySelectorAll(".view").forEach((s) => (s.hidden = s.id !== `view-${view}`));
  document.querySelectorAll("nav a").forEach((a) => (a.dataset.view === view ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
}
window.addEventListener("hashchange", route);

// ---------------- ask ----------------
function renderExamples() {
  $("#examples").innerHTML = EXAMPLES.map((q) => `<button type="button">${esc(q)}</button>`).join("");
  $("#examples").addEventListener("click", (e) => {
    if (e.target.tagName !== "BUTTON") return;
    $("#q").value = e.target.textContent;
    $("#ask-form").requestSubmit();
  });
}

$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); } });

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#q").value.trim();
  if (!question) { $("#q").focus(); return; }
  const agent = document.querySelector("input[name=agent]:checked").value;
  const btn = $("#ask-btn");
  btn.disabled = true; btn.textContent = "Researching…";
  $("#answer-col").innerHTML = `<p class="loading">Searching the reports and checking figures…</p>`;
  $("#evidence-col").innerHTML = "";
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, agent, mode: $("#offline").checked ? "offline" : "auto" }) });
    const data = await r.json().catch(() => ({ error: `The research service returned ${r.status}. Run the site with “netlify dev” or deploy it so /api/ask exists.` }));
    if (data.error) throw new Error(data.error);
    renderAnswer(data);
  } catch (err) {
    $("#answer-col").innerHTML = `<p class="error">${esc(err.message)}</p>`;
  } finally { btn.disabled = false; btn.textContent = "Ask"; }
});

function sentencesOf(text) { return text.split(/(?<=[.!?])\s+(?=[A-Z])/); }
const numsIn = (s) => (s.match(/\d[\d,]*(?:\.\d+)?/g) || []).filter((n) => !/^(\d{1,2}|20\d\d)$/.test(n));

function renderAnswer(d) {
  const citedTags = new Set(d.citations.map((c) => c.tag));
  // map each tag -> numbers in the answer sentences citing it, to highlight matching source sentences
  const tagNums = {};
  for (const s of sentencesOf(d.answer)) for (const m of s.matchAll(/\[(S\d+)\]/g)) (tagNums[m[1]] ||= new Set()), numsIn(s).forEach((n) => tagNums[m[1]].add(n));
  const answerHTML = esc(d.answer).replace(/\[(S\d+)\]/g, (_, t) => `<button class="chip" data-tag="${t}" aria-label="Show source ${t}">${t}</button>`);
  const u = d.usage || {}, ev = d.evaluation || {};
  const checks = [
    [`Citation coverage ${pct(ev.citation_coverage ?? 0)}`, ev.citation_coverage < 1],
    [`Figures grounded ${pct(ev.numeric_grounding ?? 0)}`, ev.numeric_grounding < 1],
  ];
  $("#answer-col").innerHTML = `
    <p class="answer-text">${answerHTML}</p>
    <div class="checks">${checks.map(([t, bad]) => `<span class="pill ${bad ? "bad" : ""}">${esc(t)}</span>`).join("")}</div>
    <div class="meter">
      <span>Model <b>${esc(d.model)}</b></span>
      <span>Latency <b>${d.latency_ms} ms</b></span>
      <span>Tokens <b>${u.input_tokens ?? 0} in / ${u.output_tokens ?? 0} out</b></span>
      <span>LLM calls <b>${u.calls ?? 0}</b></span>
      <span>Cost <b>$${(u.cost_usd ?? 0).toFixed(5)}</b></span>
    </div>
    ${renderTrace(d)}`;
  $("#evidence-col").innerHTML = `<h2>Source passages</h2>` + d.sources.map((s) => {
    const cited = citedTags.has(s.tag), nums = tagNums[s.tag] || new Set();
    const body = sentencesOf(s.text).map((sent) => cited && numsIn(sent).some((n) => nums.has(n)) ? `<mark>${esc(sent)}</mark>` : esc(sent)).join(" ");
    return `<section class="passage ${cited ? "" : "dim"}" id="src-${s.tag}">
      <header><b class="${cited ? "" : "uncited"}">${s.tag}</b><span>${esc(s.title)} · ${esc(s.section)}</span></header>
      <p>${body}</p></section>`;
  }).join("");
}

function renderTrace(d) {
  const t = d.trace;
  if (!t) return "";
  if (t.steps) {
    return `<section class="trace"><h2>Plan (${esc(t.planner)})</h2><ol>${t.steps.map((s) => {
      let out = "";
      if (s.tool === "extract_metric") out = `${esc(s.output.fund)}: ${s.output.value ?? "not reported"}${s.output.unit === "x" ? "x" : s.output.unit === "%" ? "%" : s.output.unit === "m" ? "m" : ""}`;
      else if (s.tool === "compare") out = `Ranked ${s.output.ranking.length} funds by ${esc((FIELD_LABELS[s.metric] || s.metric).replace(/ \(.*\)$/, ""))}`;
      else out = esc((s.output.answer || "").replace(/\s*\[S\d+\]/g, "").slice(0, 160));
      return `<li><code>${esc(s.tool)}</code> ${s.fund ? esc(s.fund) : ""} <span class="sub">${s.latency_ms} ms</span><div class="sub">${out}</div></li>`;
    }).join("")}</ol></section>`;
  }
  if (t.iterations) {
    return `<section class="trace"><h2>Self-check rounds</h2><ol>${t.iterations.map((i) =>
      `<li>Score <b>${i.score.toFixed(2)}</b> with ${i.k} passages <div class="sub">${i.issues.length ? esc(i.issues.join("; ")) : "All checks passed"}</div></li>`).join("")}</ol></section>`;
  }
  return "";
}

document.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  const el = document.getElementById(`src-${chip.dataset.tag}`);
  if (!el) return;
  el.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  el.classList.remove("pulse"); void el.offsetWidth; el.classList.add("pulse");
  setTimeout(() => el.classList.remove("pulse"), 1400);
});

// ---------------- review (human approval) ----------------
function loadReviews() { try { return JSON.parse(localStorage.getItem("pmra.reviews") || "{}"); } catch { return {}; } }
function saveReviews() { try { localStorage.setItem("pmra.reviews", JSON.stringify(state.reviews)); } catch {} updatePending(); }
function updatePending() {
  const n = state.extractions.filter((x) => !state.reviews[x.doc_id] || state.reviews[x.doc_id].status === "pending").length;
  $("#pending-count").textContent = n || "";
}
try { $("#reviewer").value = localStorage.getItem("pmra.reviewer") || ""; } catch {}
$("#reviewer").addEventListener("change", (e) => { try { localStorage.setItem("pmra.reviewer", e.target.value); } catch {} });

function renderReview() {
  const sorted = [...state.extractions].sort((a, b) => (b.checks.some((c) => c.startsWith("FAIL")) - a.checks.some((c) => c.startsWith("FAIL"))));
  $("#review-list").innerHTML = sorted.map((m) => {
    const rv = state.reviews[m.doc_id] || { status: "pending", edits: {} };
    const flagged = m.checks.some((c) => c.startsWith("FAIL"));
    const rows = Object.keys(FIELD_LABELS).map((f) => {
      const fld = m[f] || {}, edited = rv.edits && f in rv.edits;
      const v = edited ? rv.edits[f] : fld.value;
      return `<tr><td>${FIELD_LABELS[f]}</td>
        <td class="num"><input data-doc="${esc(m.doc_id)}" data-field="${f}" value="${v ?? ""}" inputmode="decimal" class="${edited ? "edited" : ""}" aria-label="${FIELD_LABELS[f]}" ${rv.status !== "pending" ? "disabled" : ""}></td>
        <td>${fld.evidence_text ? `<q>${esc(fld.evidence_text)}</q>` : `<span class="note">Not stated in report</span>`}</td></tr>`;
    }).join("");
    return `<article class="record ${flagged ? "flagged" : ""} ${rv.status}">
      <div class="record-head"><h3>${esc(m.fund_name)}</h3>
        <span>${esc(m.strategy)} · ${m.vintage} vintage · as of ${esc(m.as_of)} · ${esc(m.currency)} · extracted by ${esc(m.model_version)}</span></div>
      <div class="checks">${m.checks.map((c) => `<span class="pill ${c.startsWith("FAIL") ? "bad" : ""}">${esc(c)}</span>`).join("")}</div>
      <div class="table-wrap"><table><thead><tr><th>Field</th><th class="num">Value</th><th>Evidence</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${rv.status === "pending" ? `<div class="decision">
        <input placeholder="Note for the audit trail (optional)" data-note="${esc(m.doc_id)}">
        <button class="primary" data-approve="${esc(m.doc_id)}">Approve record</button>
        <button class="ghost reject" data-reject="${esc(m.doc_id)}">Reject record</button></div>`
      : `<p class="stamp">${rv.status === "approved" ? "Approved" : "Rejected"} by ${esc(rv.reviewer)} on ${new Date(rv.decided_at).toLocaleString()}${rv.notes ? ` — “${esc(rv.notes)}”` : ""}${Object.keys(rv.edits || {}).length ? ` · ${Object.keys(rv.edits).length} field(s) corrected` : ""}
        <button class="ghost" data-undo="${esc(m.doc_id)}">Reopen</button></p>`}
    </article>`;
  }).join("");
  updatePending();
}

$("#review-list").addEventListener("input", (e) => {
  const inp = e.target.closest("input[data-field]");
  if (!inp) return;
  const rv = (state.reviews[inp.dataset.doc] ||= { status: "pending", edits: {} });
  const orig = state.extractions.find((x) => x.doc_id === inp.dataset.doc)[inp.dataset.field]?.value;
  const v = inp.value.trim() === "" ? null : Number(inp.value);
  if (v === orig || (Number.isNaN(v))) delete rv.edits[inp.dataset.field]; else rv.edits[inp.dataset.field] = v;
  inp.classList.toggle("edited", inp.dataset.field in rv.edits);
  saveReviews();
});

$("#review-list").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const doc = b.dataset.approve || b.dataset.reject || b.dataset.undo;
  if (!doc) return;
  if (b.dataset.undo) { state.reviews[doc].status = "pending"; saveReviews(); renderReview(); return; }
  const reviewer = $("#reviewer").value.trim();
  if (!reviewer) { $("#reviewer").focus(); toast("Add your name before approving or rejecting."); return; }
  const rv = (state.reviews[doc] ||= { edits: {} });
  Object.assign(rv, { status: b.dataset.approve ? "approved" : "rejected", reviewer,
    notes: $(`[data-note="${CSS.escape(doc)}"]`)?.value || "", decided_at: new Date().toISOString() });
  saveReviews(); renderReview();
  toast(b.dataset.approve ? "Record approved" : "Record rejected");
});

$("#export-approved").addEventListener("click", () => {
  const out = state.extractions.filter((m) => state.reviews[m.doc_id]?.status === "approved").map((m) => {
    const rv = state.reviews[m.doc_id], rec = structuredClone(m);
    for (const [f, v] of Object.entries(rv.edits || {})) rec[f] = { ...rec[f], value: v, corrected_by_reviewer: true };
    return { ...rec, review: { reviewer: rv.reviewer, notes: rv.notes, decided_at: rv.decided_at } };
  });
  if (!out.length) { toast("Approve at least one record first."); return; }
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" })), download: "approved_fund_metrics.json" });
  a.click(); URL.revokeObjectURL(a.href);
});

$("#rerun-extract").addEventListener("click", async (e) => {
  e.target.disabled = true; e.target.textContent = "Extracting…";
  try {
    const fresh = await Promise.all(state.extractions.map((m) => fetch("/api/extract", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ doc_id: m.doc_id }) }).then((r) => r.json())));
    if (fresh.some((x) => x.error)) throw new Error(fresh.find((x) => x.error).error);
    state.extractions = fresh; renderReview(); toast(`Re-extracted with ${fresh[0].model_version}`);
  } catch (err) { toast("Extraction service unavailable: " + err.message); }
  finally { e.target.disabled = false; e.target.textContent = "Re-extract all"; }
});

// ---------------- benchmarks ----------------
function renderBench() {
  const sets = Object.entries(state.benchmarks);
  if (!sets.length) { $("#bench").innerHTML = `<p class="empty">No benchmark results yet. Run <code>python -m pmra.cli bench</code> then <code>python -m pmra.cli export</code>.</p>`; return; }
  $("#bench-meta").textContent = `${sets.length} model configuration(s). Dataset hash ${sets[0][1].dataset_hash}. Test split only; training questions are held out.`;
  $("#bench").innerHTML = sets.map(([name, b]) => {
    const k = 5;
    const ret = b.retrieval.map((r) => `<tr><td>${esc(r.retriever)}</td>
      <td class="num"><span class="bar" style="width:${r[`recall@${k}`] * 60}px"></span>${fmt3(r[`recall@${k}`])}</td>
      <td class="num">${fmt3(r.mrr)}</td><td class="num">${fmt3(r[`ndcg@${k}`])}</td><td class="num">${fmt3(r["hit@1"])}</td>
      <td class="num">${r.latency_ms_p50.toFixed(2)}</td><td class="num">${r.latency_ms_p95.toFixed(2)}</td></tr>`).join("");
    const ans = b.answers.map((a) => `<tr><td>${esc({ rag: "Single pass", planner: "Plan and execute", refine: "Draft and self-check" }[a.agent] || a.agent)}</td>
      <td class="num"><span class="bar" style="width:${a.accuracy * 60}px"></span>${pct(a.accuracy)}</td>
      <td class="num">${pct(a.accuracy_lookup ?? 0)}</td><td class="num">${pct(a.accuracy_compare ?? 0)}</td>
      <td class="num">${fmt3(a.citation_precision)}</td><td class="num">${(a.latency_s_p50 * 1000).toFixed(1)}</td><td class="num">${(a.latency_s_p95 * 1000).toFixed(1)}</td>
      <td class="num">${Math.round(a.tokens_in_avg)} / ${Math.round(a.tokens_out_avg)}</td><td class="num">$${a.cost_usd_per_query.toFixed(5)}</td></tr>`).join("");
    const misses = b.answers.flatMap((a) => a.rows.filter((r) => !r.correct).map((r) => `<li><b>${esc(a.agent)}</b> · ${esc(r.id)}: <span class="note">${esc(r.answer.slice(0, 180))}</span></li>`)).join("");
    const ex = b.extraction;
    const exRows = Object.entries(ex.per_field).map(([f, v]) => `<tr><td>${FIELD_LABELS[f] || f}</td><td class="num"><span class="bar" style="width:${v * 60}px"></span>${pct(v)}</td></tr>`).join("");
    return `
    <section class="block"><h2>Retrieval quality</h2>
      <p>Relevance is judged at the section level against hand-labelled gold passages. Fund routing pins results to funds named in the question.</p>
      <div class="table-wrap"><table><thead><tr><th>Retriever</th><th class="num">Recall@${k}</th><th class="num">MRR</th><th class="num">nDCG@${k}</th><th class="num">Hit@1</th><th class="num">p50 ms</th><th class="num">p95 ms</th></tr></thead><tbody>${ret}</tbody></table></div></section>
    <section class="block"><h2>Answer accuracy and cost — ${esc(name)}</h2>
      <p>An answer counts as correct only if it contains every gold figure. Citation precision is the share of cited passages that come from the right fund report.</p>
      <div class="table-wrap"><table><thead><tr><th>Agent</th><th class="num">Accuracy</th><th class="num">Lookup</th><th class="num">Comparison</th><th class="num">Cite precision</th><th class="num">p50 ms</th><th class="num">p95 ms</th><th class="num">Tokens in / out</th><th class="num">Cost / query</th></tr></thead><tbody>${ans}</tbody></table></div>
      ${misses ? `<p class="note" style="margin-top:12px">Misses</p><ul class="note">${misses}</ul>` : ""}</section>
    <section class="block"><h2>Structured extraction</h2>
      <p>Field-level accuracy against hand-verified values for ${ex.docs} reports (within 0.5%). Absent values must be returned as empty, not guessed.</p>
      <div class="table-wrap"><table><thead><tr><th>Field</th><th class="num">Accuracy</th></tr></thead><tbody>${exRows}</tbody></table></div>
      <p class="note" style="margin-top:10px">Overall ${pct(ex.field_accuracy)} · ${ex.latency_s_total.toFixed(3)} s total · $${ex.cost_usd_total.toFixed(4)}</p></section>`;
  }).join("");
}

// ---------------- models ----------------
function renderModels() {
  const { models, runs, qlora_comparison: q } = state.registry;
  const mRows = models.map((m) => `<tr><td><code>${esc(m.version_id)}</code></td><td>${esc(m.kind)}</td><td>${esc(m.base)}</td><td>${new Date(m.created_at).toLocaleString()}</td></tr>`).join("");
  const keyMetric = (r) => { const m = r.metrics; for (const k of ["accuracy", "recall@5", "field_accuracy", "eval_loss"]) if (k in m) return `${k} ${fmt3(m[k])}`; return ""; };
  const rRows = runs.slice().reverse().map((r) => `<tr><td><code>${esc(r.run_id)}</code></td><td>${esc(r.suite)}</td><td><code>${esc(r.version_id)}</code></td><td>${esc(r.config.retriever || r.config.agent || r.config.task || "")}</td><td class="num">${keyMetric(r)}</td><td><code>${esc(r.dataset_hash)}</code></td><td>${esc(r.git_sha)}</td></tr>`).join("");
  const qBlock = q ? `<div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Field accuracy</th><th class="num">Valid JSON</th><th class="num">Latency (s)</th><th class="num">Tokens in / out</th></tr></thead><tbody>${Object.entries(q).map(([k, v]) => `<tr><td><code>${esc(k)}</code></td><td class="num">${pct(v.field_accuracy)}</td><td class="num">${pct(v.json_valid_rate)}</td><td class="num">${v.latency_s_avg.toFixed(2)}</td><td class="num">${v.tokens_in} / ${v.tokens_out}</td></tr>`).join("")}</tbody></table></div>`
    : `<p>No fine-tuning run has been recorded yet. Training needs a CUDA GPU (a free Colab T4 works for a 0.5B model):</p>
       <pre class="cmd">pip install -r requirements-finetune.txt
python -m pmra.finetune.build_dataset
python -m pmra.finetune.qlora_train --base Qwen/Qwen2.5-0.5B-Instruct
python -m pmra.finetune.eval_ft --base Qwen/Qwen2.5-0.5B-Instruct --adapter artifacts/adapters/&lt;run&gt;
python -m pmra.cli export</pre><p class="note">The comparison table appears here after the last command.</p>`;
  $("#models").innerHTML = `
    <section class="block"><h2>QLoRA experiment</h2><p>Base model against its 4-bit LoRA adapter on the same held-out reports. Training data is synthetic, so none of these reports were seen in training.</p>${qBlock}</section>
    <section class="block"><h2>Model versions</h2><div class="table-wrap"><table><thead><tr><th>Version</th><th>Kind</th><th>Base</th><th>Registered</th></tr></thead><tbody>${mRows}</tbody></table></div></section>
    <section class="block"><h2>Evaluation runs</h2><div class="table-wrap"><table><thead><tr><th>Run</th><th>Suite</th><th>Model version</th><th>Config</th><th class="num">Headline metric</th><th>Dataset</th><th>Commit</th></tr></thead><tbody>${rRows}</tbody></table></div></section>`;
}

// ---------------- boot ----------------
(async function boot() {
  renderExamples(); route();
  const [index, ext, bench, reg] = await Promise.allSettled(["data/index.json", "data/extractions.json", "data/benchmarks.json", "data/registry.json"].map(getJSON));
  if (index.status === "fulfilled") {
    const docs = {};
    index.value.forEach((c) => (docs[c.doc_id] ||= c.meta));
    $("#corpus-list").innerHTML = Object.values(docs).map((m) => `<li>${esc(m.fund)}<small>${esc(m.strategy)} · ${esc(m.report_date)}</small></li>`).join("");
  }
  if (ext.status === "fulfilled") { state.extractions = ext.value; renderReview(); }
  if (bench.status === "fulfilled") { state.benchmarks = bench.value; renderBench(); }
  if (reg.status === "fulfilled") { state.registry = reg.value; renderModels(); }
})();
