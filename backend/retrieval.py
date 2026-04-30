from sentence_transformers import SentenceTransformer
import os

# reuse the Pinecone index object from your existing code
# from your ingest/app module: from pinecone import Pinecone; pc = Pinecone(...); index = pc.Index(INDEX_NAME)
# here assume `index` is importable or you initialize it the same way in this module

# load the same local embedding model used for ingest
_SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(question: str, index, top_k: int = 4):
    """
    Embed `question`, query Pinecone `index`, and return a list of matches.
    Each match is a dict containing: id, score, metadata, and (if available) text snippet.
    """
    # compute embedding (numpy -> list)
    qvec = _SBERT_MODEL.encode(question, convert_to_numpy=True).tolist()

    # query Pinecone
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True)

    matches = []
    for raw in res.get("matches", []):
        # support both dict-style and object-style match
        if isinstance(raw, dict):
            mid = raw.get("id")
            score = raw.get("score")
            md = raw.get("metadata") or {}
        else:
            # object-style returned by some pinecone clients
            mid = getattr(raw, "id", None)
            score = getattr(raw, "score", None)
            md = getattr(raw, "metadata", None) or {}

        # ensure metadata is a plain dict
        if not isinstance(md, dict):
            try:
                md = dict(md)
            except Exception:
                md = {}

        match = {
            "id": mid,
            "score": score,
            "metadata": md,
            "text": md.get("text") or md.get("chunk_text") or None
        }
        print("RETRIEVAL MATCH:", {"id": mid, "score": score, "md": md})
        matches.append(match)
    return matches