"""
BDE Academy RAG — core library.

Loads a prebuilt index of the handbook, retrieves the most relevant passages
(semantic + keyword hybrid), and answers questions with an LLM. Works with any
OpenAI-compatible chat endpoint (OpenAI, Groq, Together, OpenRouter, Ollama...).

If no LLM key is configured, it falls back to retrieval-only mode: it returns
the most relevant passages so the learner can still read and brainstorm.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
INDEX_DIR = Path(os.environ.get("BDE_INDEX_DIR", HERE / "data" / "index"))

EMBED_MODEL = os.environ.get("EMBED_MODEL", "minishlab/potion-base-8M")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")

_model = None
_index: Optional["Index"] = None


# ----------------------------------------------------------------------------
# Embeddings (model2vec: torch-free static embeddings)
# ----------------------------------------------------------------------------
def _get_model():
    global _model
    if _model is None:
        from model2vec import StaticModel
        _model = StaticModel.from_pretrained(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    v = _get_model().encode(texts)
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------
class Index:
    def __init__(self, chunks: list[dict], vecs: np.ndarray, manifest: dict):
        self.chunks = chunks
        self.vecs = vecs
        self.manifest = manifest
        self._build_bm25()

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "Index":
        index_dir = Path(index_dir)
        chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        vecs = np.load(index_dir / "vectors.npz")["vecs"]
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        return cls(chunks, vecs, manifest)

    def _build_bm25(self):
        self._tokens = [self._tok(f"{c.get('chapter_title','')} {c.get('section_title','')} {c['text']}") for c in self.chunks]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = np.array([len(t) for t in self._tokens], dtype=np.float32)
        self._avg_len = float(self._len.mean()) if len(self._len) else 1.0
        df = Counter()
        for t in self._tokens:
            for w in set(t):
                df[w] += 1
        n = len(self._tokens)
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    @staticmethod
    def _tok(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    def bm25(self, query: str) -> np.ndarray:
        q = self._tok(query)
        k1, b = 1.5, 0.75
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, tf in enumerate(self._tf):
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = self._idf.get(w, 0.0)
                f = tf[w]
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * self._len[i] / self._avg_len))
            scores[i] = s
        return scores

    def semantic(self, query: str) -> np.ndarray:
        qv = embed([query])[0]
        return self.vecs @ qv

    def search(self, query: str, k: int = 6, chapter: Optional[str] = None) -> list[dict]:
        """Hybrid retrieval via reciprocal rank fusion of semantic + BM25."""
        sem = self.semantic(query)
        bm = self.bm25(query)

        def ranks(scores: np.ndarray) -> dict[int, int]:
            order = np.argsort(-scores)
            return {int(i): r for r, i in enumerate(order)}

        rs, rb = ranks(sem), ranks(bm)
        kk = 60.0
        fused = {i: 1.0 / (kk + rs[i]) + 1.0 / (kk + rb[i]) for i in range(len(self.chunks))}

        order = sorted(fused, key=lambda i: -fused[i])
        out, seen = [], set()
        for i in order:
            c = self.chunks[i]
            if chapter and str(c.get("chapter")) != str(chapter):
                continue
            key = (c.get("chapter"), c.get("section")) if c.get("section") else ("id", c["id"])
            if key in seen:
                continue
            seen.add(key)
            d = dict(c)
            d["score"] = round(float(sem[i]), 4)
            d["rrf"] = round(float(fused[i]), 6)
            out.append(d)
            if len(out) >= k:
                break
        return out


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index.load()
    return _index


# ----------------------------------------------------------------------------
# Prompting
# ----------------------------------------------------------------------------
MODES = {
    "ask": "Answer the question using ONLY the handbook context provided. Be accurate and concise. "
           "If the context does not cover it, say so plainly instead of inventing. Cite the chapter/section.",
    "explain": "Explain the topic as if the reader is brand new to IT and to sales. Use a simple analogy, "
               "short sentences, and a concrete example. Avoid jargon; define any term you must use.",
    "brainstorm": "Help the learner brainstorm. Offer several distinct ideas, angles, questions to ask, "
                  "or ways to approach the situation. Ground every idea in the handbook context and label them clearly.",
    "quiz": "Create 5 practice questions from the context, mixing recall and applied scenarios. "
            "Give the answers at the end with a one-line explanation each.",
    "roleplay": "Roleplay as the client named or implied by the learner. Stay in character, be realistic and "
                "reasonably sceptical, and let the learner practise. At the end, step out of character and give "
                "specific coaching feedback.",
    "coach": "Act as a study coach. Build a short, ordered plan from the context: what to learn first, what to "
             "practise, and one task to complete. Keep it practical.",
}

STARTERS = [
    "What does an IT BDE actually do day to day?",
    "Explain what discovery is, simply.",
    "Brainstorm 5 ways to research a prospect before outreach.",
    "Quiz me on the software development lifecycle.",
    "Roleplay a CTO who says 'we already have a team'.",
    "Build me a one-week study plan to start.",
]


def _context_block(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        head = f"[Chapter {c.get('chapter','?')}: {c.get('chapter_title','')} \u2014 \u00a7{c.get('section','')} {c.get('section_title','')}]"
        parts.append(f"{head}\n{c['text'].strip()}")
    return "\n\n---\n\n".join(parts)


def _call_llm(system: str, user: str, temperature: float = 0.3) -> str:
    if not LLM_API_KEY:
        return ""
    body = json.dumps({
        "model": LLM_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        return f"[LLM error {e.code}] {detail}"
    except Exception as e:  # noqa: BLE001
        return f"[LLM error] {e}"


def answer(query: str, mode: str = "ask", k: int = 6, chapter: Optional[str] = None,
           temperature: float = 0.3) -> dict:
    """Retrieve and answer. Returns {answer, sources, mode, used_llm}."""
    idx = get_index()
    chunks = idx.search(query, k=k, chapter=chapter)
    context = _context_block(chunks)
    system = (
        "You are the Easenext BDE Academy coach. You help a new Business Development Executive "
        "learn and think using their own IT-BDE handbook. Be practical, warm and plain-spoken. "
        + MODES.get(mode, MODES["ask"])
    )
    user = f"Handbook context:\n\n{context}\n\n---\n\nLearner's request:\n{query}"

    text = _call_llm(system, user, temperature=temperature)
    used_llm = bool(text)
    if not text:
        # Retrieval-only fallback: still genuinely useful.
        lines = ["**Most relevant passages from the handbook** "
                 "(no LLM key configured \u2014 showing the source text so you can read and think directly):\n"]
        for c in chunks:
            lines.append(f"**Ch.{c.get('chapter','?')} {c.get('chapter_title','')} \u2014 "
                         f"\u00a7{c.get('section','')} {c.get('section_title','')}**\n{c['text'].strip()}")
        text = "\n\n".join(lines)

    return {"answer": text, "sources": chunks, "mode": mode, "used_llm": used_llm}


def source_label(c: dict) -> str:
    return f"Ch.{c.get('chapter','?')} {c.get('chapter_title','')} \u2014 \u00a7{c.get('section','')} {c.get('section_title','')} (p.{c.get('page','?')})"


def chapters() -> list[tuple]:
    idx = get_index()
    seen = {}
    for c in idx.chunks:
        n = c.get("chapter")
        if n not in seen:
            seen[n] = c.get("chapter_title", "")
    return sorted(seen.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 999)
