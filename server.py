"""Easenext BDE Academy — hosted web app (FastAPI).

Serves a single-page chat UI at "/" and a small JSON API under /api.
Deployable to any container host (Hugging Face Spaces, Render, Railway, Fly.io).

Env vars:
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL   # any OpenAI-compatible endpoint
  APP_PASSWORD                           # optional: gate the app behind a password
  PORT                                   # port to listen on (default 7860)
"""
import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import rag

HERE = Path(__file__).resolve().parent
APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()

app = FastAPI(title="Easenext BDE Academy")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {
        "llm": bool(os.environ.get("LLM_API_KEY")),
        "model": os.environ.get("LLM_MODEL", ""),
        "password_required": bool(APP_PASSWORD),
        "chunks": rag.get_index().manifest.get("chunks"),
        "chapters": len(rag.chapters()),
    }


@app.get("/api/chapters")
def chapters():
    return [{"n": str(n), "t": t} for n, t in rag.chapters()]


@app.post("/api/ask")
async def ask(req: Request):
    body = await req.json()
    if APP_PASSWORD and str(body.get("password", "")) != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    result = rag.answer(
        query,
        mode=body.get("mode", "ask"),
        k=int(body.get("k", 6)),
        chapter=body.get("chapter") or None,
    )
    return {
        "answer": result["answer"],
        "used_llm": result["used_llm"],
        "sources": [{"label": rag.source_label(c)} for c in result["sources"]],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
