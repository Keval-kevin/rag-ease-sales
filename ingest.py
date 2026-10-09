"""
BDE Academy RAG — ingestion.

Turns the handbook PDF into a searchable index:
  PDF -> pages -> chapters/sections (structure-aware) -> chunks -> embeddings.

Run:
  python ingest.py --pdf data/handbook.pdf --out data/index
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pypdf
from model2vec import StaticModel

HEADER = re.compile(r"IT BUSINESS DEVELOPMENT EXECUTIVE HANDBOOK", re.I)
PAGE = re.compile(r"^\s*Page\s+\d+\s*$", re.I)
CHAP = re.compile(r"^(\d{1,2})\s+([A-Z].{2,70})$")
SECT = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+(.{2,80})$")

SKIP_PAGES = 3  # cover, how-to-use, contents


def extract_items(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a flat list of (page_number, line) with running headers/footers removed."""
    reader = pypdf.PdfReader(str(pdf_path))
    items: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        if i < SKIP_PAGES:
            continue
        text = page.extract_text() or ""
        for raw in text.split("\n"):
            s = raw.rstrip()
            if not s.strip():
                continue
            if HEADER.search(s) or PAGE.match(s):
                continue
            items.append((i + 1, s))
    return items


def find_chapters(items: list[tuple[int, str]]) -> list[tuple[int, str, int]]:
    """Detect chapter starts in order. A heading counts if it is the next chapter
    in sequence AND (its first sub-section follows soon, or it starts a new page)."""
    starts = []
    expected = 1
    for j, (pg, line) in enumerate(items):
        m = CHAP.match(line.strip())
        if not m:
            continue
        n = int(m.group(1))
        if n != expected or not (1 <= n <= 40):
            continue
        first_on_page = (j == 0) or (items[j - 1][0] != pg)
        has_subsection = False
        for k in range(j + 1, min(j + 6, len(items))):
            ms = SECT.match(items[k][1].strip())
            if ms and int(ms.group(1)) == n and int(ms.group(2)) <= 2:
                has_subsection = True
                break
        if has_subsection or first_on_page:
            starts.append((n, m.group(2).strip(), j))
            expected = n + 1
    return starts


def is_section(line: str, chapter: int) -> re.Match | None:
    m = SECT.match(line)
    if not m:
        return None
    if int(m.group(1)) != chapter or int(m.group(2)) > 40:
        return None
    title = m.group(3).strip()
    if not title or not title[0].isupper() or len(line) > 82:
        return None
    return m


def build_sections(items, starts):
    sections = []
    bounds = [s[2] for s in starts] + [len(items)]
    for idx, (n, title, j) in enumerate(starts):
        cur = None
        for k in range(j, bounds[idx + 1]):
            pg, raw = items[k]
            s = raw.strip()
            if s == f"{n} {title}":
                continue
            m = is_section(s, n)
            if m:
                if cur:
                    sections.append(cur)
                cur = {"chapter": n, "chapter_title": title,
                       "section": f"{m.group(1)}.{m.group(2)}", "section_title": m.group(3).strip(),
                       "page": pg, "lines": []}
            else:
                if cur is None:
                    cur = {"chapter": n, "chapter_title": title, "section": "",
                           "section_title": "", "page": pg, "lines": []}
                cur["lines"].append(s)
        if cur:
            sections.append(cur)
    for s in sections:
        s["text"] = re.sub(r"\s+", " ", " ".join(s["lines"])).strip()
        del s["lines"]
    return [s for s in sections if len(s["text"]) > 40]


def chunk(text: str, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, buf = [], ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 > max_chars and buf:
            out.append(buf.strip())
            buf = buf[-overlap:] + " " if overlap else ""
        buf += sent + " "
    if buf.strip():
        out.append(buf.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/handbook.pdf")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--embed-model", default="minishlab/potion-base-8M")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {pdf_path} ...")
    items = extract_items(pdf_path)
    starts = find_chapters(items)
    print(f"  {len(items)} lines, {len(starts)} chapters detected")
    sections = build_sections(items, starts)
    print(f"  {len(sections)} sections")

    chunks = []
    for s in sections:
        for part in chunk(s["text"]):
            chunks.append({
                "id": len(chunks),
                "chapter": s["chapter"], "chapter_title": s["chapter_title"],
                "section": s["section"], "section_title": s["section_title"],
                "page": s["page"], "text": part,
            })
    print(f"  {len(chunks)} chunks")

    print(f"Embedding with {args.embed_model} ...")
    model = StaticModel.from_pretrained(args.embed_model)
    corpus = [f"{c['chapter_title']} — {c['section_title']}. {c['text']}" for c in chunks]
    vecs = np.asarray(model.encode(corpus), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    np.savez(out / "vectors.npz", vecs=vecs)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "source": pdf_path.name, "embed_model": args.embed_model,
        "dim": int(vecs.shape[1]), "chunks": len(chunks),
        "chapters": len(starts),
    }, indent=2), encoding="utf-8")
    print(f"Done. Index written to {out}/ ({vecs.shape})")


if __name__ == "__main__":
    main()
