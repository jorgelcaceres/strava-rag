# Strava RAG — README

Short description
- Strava RAG is a small Retrieval‑Augmented Generation prototype that lets you ask natural language questions about your Strava activities.  
- Stack: Python, FastAPI backend, Streamlit frontend, local SBERT embeddings (sentence‑transformers), Pinecone vector DB, OpenAI Chat for answer generation.

Quick status
- Ingest: strava_fetch.py + ingest.py (fetch activities, chunk, embed locally, upsert to Pinecone)
- Query: FastAPI endpoint /query (retrieve top_k matches, build prompt, call OpenAI, return answer)
- UI: Streamlit app (frontend/streamlit_app.py) — spinner and persisted history

Prerequisites
- Python 3.10+ (venv recommended)
- Node/npm only if you choose React frontend (optional)
- Pinecone account (free tier OK for small datasets)
- OpenAI account + API key (for chat completions)
- Strava developer app credentials (CLIENT_ID, CLIENT_SECRET, refresh token)
- Recommended packages: see requirements.txt

Environment variables (.env)
Create a .env file at the project root (do NOT commit it). Required keys:
- OPENAI_API_KEY=sk-...
- PINECONE_API_KEY=pcsk-...
- PINECONE_ENV=us-west1-gcp        # example, set to the Pinecone env from console
- PINECONE_INDEX=strava-rag
- STRAVA_CLIENT_ID=...
- STRAVA_CLIENT_SECRET=...
- STRAVA_REFRESH_TOKEN=...

Install (one-time)
- Create and activate a venv:
  python3 -m venv venv
  source venv/bin/activate
- Install deps:
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install -r requirements.txt

Project layout (important files)
- backend/
  - app.py              — FastAPI server (POST /query)
  - ingest.py           — fetch + chunk + embed + upsert
  - retrieval.py        — query Pinecone and normalize matches
  - llm.py              — build_prompt and OpenAI call
  - __init__.py
- frontend/
  - streamlit_app.py    — Streamlit UI (spinner + persisted history)
- .env
- requirements.txt

Create Pinecone index (if needed)
- The backend will create the index programmatically if missing (ensure PINECONE_API_KEY and PINECONE_ENV are set). Index dimension must match embedding model (default: 384 for all‑MiniLM‑L6‑v2).

Ingest Strava data (one-time / periodically)
1. Ensure .env contains STRAVA_CLIENT_* and REFRESH_TOKEN.
2. Run:
   source venv/bin/activate
   python backend/ingest.py
- This fetches activities via the Strava API, splits into chunks, computes embeddings locally (sentence‑transformers), and upserts vectors + metadata to Pinecone.
- If you re-create the index or change embedding model/dimension, delete the old index in Pinecone Console or use a new PINECONE_INDEX name.

Run the backend (server)
- From project root:
  source venv/bin/activate
  uvicorn backend.app:app --reload --port 8000 --host 127.0.0.1
- Test:
  curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" \
    -d '{"question":"How many times did I go to Sausalito last year?","top_k":4}'

Run the Streamlit UI
- From project root:
  source venv/bin/activate
  BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
- The UI persists conversation to frontend/history.json and shows a spinner while backend responds. To show sources, include phrases like "show me the source" in your prompt.

Costs & notes
- Embeddings: using local sentence-transformers in this prototype → no OpenAI embedding cost.
- Pinecone: free tier has limited storage/QPS; check Pinecone Console for usage.
- OpenAI: chat completions (gpt-3.5-turbo) billed per token (prompt + completion). Use tiktoken to measure token counts and tune top_k / snippet lengths to reduce cost.
- Security: never commit .env or secrets. For production use secret managers (AWS/GCP/Azure).

Troubleshooting tips
- "insufficient_quota" / 429 from OpenAI: add billing/payment method or reduce model usage.
- Pinecone errors: check PINECONE_API_KEY, PINECONE_ENV, index dimension, and usage quotas.
- Virtualenv/VS Code: ensure the correct interpreter is selected in VS Code and the venv is activated in terminals.
- If retrieval returns no matches: verify index contains vectors and retrieval uses same embedding model as ingest.
- If LLM responds "I don't know": ensure chunk text or relevant metadata (distance, date) is included in build_prompt.

Recommended next steps
- Include more metadata (distance, time) in chunk text so LLM has explicit facts.
- Add re-ranking step or a verifier model to reduce hallucination.
- Add conversation memory (store last N turns) and streaming responses for better UX.
- Consider switching vector store to pgvector if Pinecone free tier is limiting.
