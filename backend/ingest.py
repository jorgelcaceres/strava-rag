from pinecone import Pinecone, ServerlessSpec
import os
import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from strava_fetch import get_access_token, fetch_activities

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "strava-rag")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use all-MiniLM-L6-v2 (embedding dim = 384)
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Create index if missing (matches local model dim)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region=PINECONE_ENV or "us-west1-gcp")
    )

index = pc.Index(INDEX_NAME)

# Load local embedding model
sbert = SentenceTransformer(MODEL_NAME)

def activity_to_docs(acts):
    docs = []
    for a in acts:
        text = (
            f"Name: {a.get('name')}\n"
            f"Date: {a.get('start_date_local')}\n"
            f"Type: {a.get('type')}\n"
            f"Distance_m: {a.get('distance')}\n"
            f"Elapsed_s: {a.get('elapsed_time')}\n"
            f"Avg_speed: {a.get('average_speed')}\n"
            f"Description: {a.get('description') or ''}\n"
        )
        docs.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": {
                "activity_id": a.get("id"),
                "type": a.get("type"),
                "name": a.get("name"),
                "date": a.get("start_date_local")
            }
        })
    return docs

def ingest_all():
    token = get_access_token()
    acts = fetch_activities(token)
    docs = activity_to_docs(acts)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for d in docs:
        parts = splitter.split_text(d["text"])
        for i, p in enumerate(parts):
            chunks.append({
                "id": f"{d['id']}-{i}",
                "text": p,
                "metadata": d["metadata"]
            })

    # Batch encode and upsert
    ids_batch = []
    texts_batch = []
    metas_batch = []

    for c in chunks:
        ids_batch.append(c["id"])
        texts_batch.append(c["text"])
        metas_batch.append(c["metadata"])

        if len(texts_batch) >= 64:
            vectors = sbert.encode(texts_batch, convert_to_numpy=True).tolist()
            to_upsert = [(ids_batch[i], vectors[i], metas_batch[i]) for i in range(len(ids_batch))]
            index.upsert(vectors=to_upsert)
            ids_batch, texts_batch, metas_batch = [], [], []

    # final flush
    if texts_batch:
        vectors = sbert.encode(texts_batch, convert_to_numpy=True).tolist()
        to_upsert = [(ids_batch[i], vectors[i], metas_batch[i]) for i in range(len(ids_batch))]
        index.upsert(vectors=to_upsert)

    print(f"Upserted {len(chunks)} chunks to Pinecone index '{INDEX_NAME}'")

if __name__ == "__main__":
    ingest_all()
