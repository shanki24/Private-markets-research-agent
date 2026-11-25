"""Ingest fund reports (Markdown, text, PDF) into section-aware, overlapping chunks."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    page: int | None = None
    meta: dict = field(default_factory=dict)


FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader  # optional dependency

    reader = PdfReader(str(path))
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def _sections(body: str) -> list[tuple[str, str]]:
    """Split Markdown on headings; returns (section_title, text)."""
    out, current, buf = [], "Preamble", []
    for line in body.splitlines():
        h = re.match(r"^#{1,6}\s+(.*)", line)
        if h:
            if buf and "".join(buf).strip():
                out.append((current, "\n".join(buf).strip()))
            current, buf = h.group(1).strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        out.append((current, "\n".join(buf).strip()))
    return out


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-aligned sliding window of roughly `size` words."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= size:
            chunks.append(" ".join(cur))
            # carry trailing sentences worth ~overlap words
            carry, n = [], 0
            for x in reversed(cur):
                n += len(x.split())
                carry.insert(0, x)
                if n >= overlap:
                    break
            cur = carry if len(carry) < len(cur) else []
    if cur:
        tail = " ".join(cur)
        if not chunks or tail not in chunks[-1]:
            chunks.append(tail)
    return chunks


def ingest_file(path: Path) -> list[Chunk]:
    if path.suffix.lower() == ".pdf":
        pages = _read_pdf(path)
        meta = {"source_file": path.name}
        doc_id, title = path.stem, path.stem.replace("_", " ")
        units = [(f"Page {p}", t, p) for p, t in pages]
    else:
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        meta["source_file"] = path.name
        doc_id = meta.get("doc_id", path.stem)
        t = re.search(r"^#\s+(.*)", body, re.M)
        title = t.group(1).strip() if t else path.stem
        units = [(s, txt, None) for s, txt in _sections(body) if s != title]

    chunks: list[Chunk] = []
    for section, text, page in units:
        for piece in _window(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP):
            chunks.append(Chunk(f"{doc_id}#c{len(chunks)}", doc_id, title, section, piece, page, meta))
    return chunks


def ingest_dir(directory: Path = config.REPORTS_DIR, out: Path = config.CHUNKS_FILE) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in {".md", ".txt", ".pdf"}:
            chunks.extend(ingest_file(p))
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    return chunks


def load_chunks(path: Path = config.CHUNKS_FILE) -> list[Chunk]:
    if not path.exists():
        return ingest_dir()
    return [Chunk(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
