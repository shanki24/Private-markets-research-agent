// Anthropic Messages API client with token + cost accounting.
// Prices are placeholders (USD per million tokens); set real ones via env vars.
export const MODEL = process.env.PMRA_MODEL || "claude-sonnet-5-5";
const PRICE_IN = parseFloat(process.env.PRICE_IN_PER_MTOK || "3");
const PRICE_OUT = parseFloat(process.env.PRICE_OUT_PER_MTOK || "15");

export const hasKey = () => Boolean(process.env.ANTHROPIC_API_KEY);
export const newUsage = () => ({ input_tokens: 0, output_tokens: 0, cost_usd: 0, calls: 0 });
export const addUsage = (a, b) => { for (const k in b) a[k] += b[k]; return a; };
export const estTokens = (s) => Math.max(1, Math.floor(s.length / 4));

export async function complete(system, user, maxTokens = 600) {
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "x-api-key": process.env.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
      "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, max_tokens: maxTokens, temperature: 0, system,
      messages: [{ role: "user", content: user }] }),
  });
  if (!r.ok) throw new Error(`Anthropic API ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const d = await r.json();
  const text = d.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  const u = d.usage;
  return { text, usage: { input_tokens: u.input_tokens, output_tokens: u.output_tokens,
    cost_usd: (u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6, calls: 1 } };
}

export const parseJSON = (t) => JSON.parse(t.trim().replace(/^```(?:json)?/m, "").replace(/```$/m, ""));
