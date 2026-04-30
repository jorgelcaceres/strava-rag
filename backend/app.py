import os
import logging
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Pinecone client
from pinecone import Pinecone, ServerlessSpec

# retrieval helper (must exist)
from backend.retrieval import retrieve

# llm helpers
from backend.llm import build_prompt, answer_with_llm

# logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "strava-rag")
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region=PINECONE_ENV or "us-west1-gcp")
    )
index = pc.Index(INDEX_NAME)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryIn(BaseModel):
    question: str
    top_k: int = 4
    include_sources: bool = False
    max_tokens: int = 300

class QueryOut(BaseModel):
    answer: str
    sources: list
    contexts: list

@app.post("/query", response_model=QueryOut)
async def query_endpoint(payload: QueryIn):
    # retrieve
    matches = retrieve(payload.question, index, top_k=payload.top_k)

    # build prompt (no ActivityID)
    prompt = build_prompt(payload.question, matches)

    # call LLM
    answer = answer_with_llm(prompt, model="gpt-3.5-turbo", max_tokens=payload.max_tokens, temperature=0.0)

    # prepare contexts and sources (but only return them if requested)
    sources = []
    contexts = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        contexts.append({
            "id": m.get("id"),
            "metadata": md,
            "text": m.get("text")
        })
        sources.append(md.get("activity_id") or md.get("id"))

    return {
        "answer": answer,
        "sources": sources if payload.include_sources else [],
        "contexts": contexts if payload.include_sources else []
    }
