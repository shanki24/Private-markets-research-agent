// POST /api/extract {doc_id, mode}  -> structured fund metrics with evidence + validation checks
import { docChunks, extractRules, FUNDS, METRIC_FIELDS, validate } from "./lib/core.mjs";
import { complete, hasKey, MODEL, parseJSON } from "./lib/llm.mjs";

const SYSTEM = `Extract fund metrics from the report. Return ONLY JSON:
{"fund_name": str, "strategy": str, "vintage": int, "as_of": "YYYY-MM-DD", "currency": str,
 "<field>": {"value": number|null, "unit": str, "evidence_chunk": "<chunk id>", "evidence_text": "<exact short quote>"}}
for fields: ${METRIC_FIELDS.join(", ")}. Money in millions of fund currency. Percents as numbers. null if absent. Never guess.`;

export default async (req) => {
  const { doc_id, mode = "auto" } = await req.json();
  if (!FUNDS[doc_id]) return Response.json({ error: `Unknown report ${doc_id}` }, { status: 404 });
  const t0 = Date.now();
  if (mode === "offline" || !hasKey())
    return Response.json({ ...extractRules(doc_id), model_version: "offline-rules-v1", latency_ms: Date.now() - t0 });
  try {
    const body = docChunks(doc_id).map((c) => `<chunk id='${c.chunk_id}' section='${c.section}'>\n${c.text}\n</chunk>`).join("\n\n");
    const { text, usage } = await complete(SYSTEM, body, 1500);
    const m = validate({ doc_id, ...parseJSON(text) });
    return Response.json({ ...m, model_version: MODEL, usage, latency_ms: Date.now() - t0 });
  } catch (e) {
    return Response.json({ ...extractRules(doc_id), model_version: "offline-rules-v1",
      warning: `LLM extraction failed, fell back to rules: ${e.message}` });
  }
};

export const config = { path: "/api/extract" };
