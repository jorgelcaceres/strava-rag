import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_INSTRUCTION = (
    "You are an assistant that answers questions about a user's Strava activities. "
    "Use ONLY the provided activity context and cite the source entries when relevant. "
    "If the answer is not contained in the context, say you don't know."
)

def build_prompt(question: str, matches: List[Dict], max_chars_per_snippet: int = 800) -> str:
    """
    Build prompt WITHOUT ActivityID prefixes to avoid leaking raw IDs into the model prompt.
    """
    ctx_lines = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        snippet = m.get("text") or md.get("text") or md.get("summary")
        if snippet:
            snippet = snippet.strip().replace("\n", " ")
            snippet = snippet[:max_chars_per_snippet]
            distance = md.get("distance") or md.get("Distance_m") or md.get("distance_m") or md.get("Distance")
            distance_str = f" | distance: {distance} m" if distance not in (None, "", 0) else ""
            ctx_lines.append(f"{md.get('date','')} | {md.get('name','')}{distance_str}: {snippet}")
        else:
            ctx_lines.append(f"{md.get('date','')} | {md.get('name','')}")
    context = "\n\n".join(ctx_lines) if ctx_lines else "No activity context available."
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\nContext:\n{context}\n\n"
        f"Question: {question}\n\nAnswer concisely and cite the source entries when applicable."
    )
    return prompt

def _count_tokens_for_model(text: str, model_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def answer_with_llm(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 300, temperature: float = 0.0) -> str:
    logger.debug(f"[LLM] model={model} max_tokens={max_tokens} temperature={temperature}")

    prompt_tokens = _count_tokens_for_model(prompt, model)
    estimated_total = prompt_tokens + max_tokens
    logger.info(f"[LLM] Prompt tokens: {prompt_tokens}, max_tokens: {max_tokens}, estimated total tokens: {estimated_total}")

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # resp.choices[0].message is an object; access .content
    msg = resp.choices[0].message
    content = (msg.content if hasattr(msg, "content") else str(msg)).strip()

    try:
        response_tokens = _count_tokens_for_model(content, model)
        logger.info(f"[LLM] Response tokens (actual): {response_tokens}")
    except Exception:
        pass

    logger.info(f"[LLM] OpenAI response received: {resp}")
    return content
