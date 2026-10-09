---
title: Easenext BDE Academy
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Easenext BDE Academy — hosted web app

A shareable web app where anyone on the team can **ask questions, brainstorm, get quizzed
and practise roleplays** — grounded in your *IT Business Development Executive Handbook (2026)*.

- **Backend:** FastAPI (`server.py`) + a small RAG engine (`rag.py`)
- **Frontend:** one self-contained page (`static/index.html`), mobile-friendly
- **Ready to run:** the search index is already built (29 chapters · 188 sections · 288 chunks)
- **Embeddings run locally** (torch-free, ~30 MB) — no embedding cost
- **Answering** uses Groq by default (free tier, OpenAI-compatible); without a key it still works in retrieval-only mode

---

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # paste a Groq key from https://console.groq.com/keys
export $(grep -v '^#' .env | xargs)
uvicorn server:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860

Or with Docker:

```bash
docker build -t bde-academy .
docker run -p 7860:7860 --env-file .env bde-academy
```

---

## Deploy to a public link

### Option A — Hugging Face Spaces (free, easiest, ~3 min)

1. Create a Space at https://huggingface.co/new-space → **SDK: Docker** → *Blank*.
2. Upload this folder (all files, including `data/` and `static/`).
   Spaces reads the `sdk: docker` front-matter at the top of this README automatically.
3. In **Settings → Variables and secrets**, add secrets:
   `LLM_API_KEY` (Groq), `LLM_BASE_URL=https://api.groq.com/openai/v1`,
   `LLM_MODEL=openai/gpt-oss-20b`, and optionally `APP_PASSWORD`.
4. The app builds and goes live at `https://<user>-<space>.hf.space`.

### Option B — Render (free tier)

1. Push this folder to a GitHub repo.
2. Render → **New → Web Service** → connect the repo → it reads `render.yaml`.
3. Set the secret `LLM_API_KEY` (and `APP_PASSWORD` if you want it gated). Deploy.

### Option C — Railway

```bash
railway init && railway up
railway variables set LLM_API_KEY=gsk_... LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_MODEL=openai/gpt-oss-20b
```

### Option D — Fly.io

```bash
fly launch --no-deploy     # keeps the bundled fly.toml
fly secrets set LLM_API_KEY=gsk_... LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_MODEL=openai/gpt-oss-20b APP_PASSWORD=choose-one
fly deploy
```

---

## Environment variables

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible base URL (default Groq: `https://api.groq.com/openai/v1`) |
| `LLM_API_KEY` | Groq key from https://console.groq.com/keys. If unset, retrieval-only mode |
| `LLM_MODEL` | Default `openai/gpt-oss-20b` (Groq free). Also `openai/gpt-oss-120b` |
| `APP_PASSWORD` | Optional. If set, users must enter it to ask |
| `PORT` | Port to listen on (default `7860`) |
| `EMBED_MODEL` | model2vec embedding model (default `minishlab/potion-base-8M`) |

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | the chat UI |
| `/api/config` | GET | LLM status, chapter/chunk counts |
| `/api/chapters` | GET | chapter list for the filter |
| `/api/ask` | POST | `{query, mode, k, chapter, password}` → `{answer, used_llm, sources}` |
| `/healthz` | GET | health check |

`mode` is one of `ask`, `explain`, `brainstorm`, `quiz`, `roleplay`, `coach`.

---

## Add or replace documents

Put a PDF in `data/` and rebuild the index, then redeploy:

```bash
python ingest.py --pdf data/handbook.pdf --out data/index
```

---

## Notes

- Your handbook stays on the server; it is only sent to the LLM you configure.
- For a public Space, set `APP_PASSWORD` so only your team can use it.
- Costs: hosting is free on the tiers above; only the LLM calls cost money (and only if you set a key).
