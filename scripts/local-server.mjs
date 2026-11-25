// Zero-dependency local server: serves /web and routes /api/* to the Netlify functions.
// Usage: node scripts/local-server.mjs   (set ANTHROPIC_API_KEY for LLM mode)
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import ask from "../netlify/functions/ask.mjs";
import extract from "../netlify/functions/extract.mjs";

const ROOT = new URL("../web/", import.meta.url).pathname;
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml" };
const routes = { "/api/ask": ask, "/api/extract": extract };
const port = process.env.PORT || 8888;

createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${port}`);
  if (routes[url.pathname]) {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const r = await routes[url.pathname](new Request(url, { method: req.method, headers: req.headers,
      body: req.method === "POST" ? Buffer.concat(chunks) : undefined }));
    res.writeHead(r.status, { "content-type": r.headers.get("content-type") || "application/json" });
    return res.end(Buffer.from(await r.arrayBuffer()));
  }
  const path = normalize(join(ROOT, url.pathname === "/" ? "index.html" : url.pathname));
  if (!path.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
  try {
    const body = await readFile(path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404); res.end("Not found"); }
}).listen(port, () => console.log(`Research desk on http://localhost:${port}  (LLM mode: ${process.env.ANTHROPIC_API_KEY ? "on" : "offline"})`));
