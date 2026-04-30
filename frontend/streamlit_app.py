import os
import json
import requests
import streamlit as st
from pathlib import Path

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
HISTORY_PATH = Path("frontend") / "history.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Strava RAG Chat", layout="centered")

def load_history():
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history):
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

if "history" not in st.session_state:
    st.session_state.history = load_history()

st.title("Strava RAG Chat (Streamlit)")

def wants_sources(q: str) -> bool:
    ql = q.lower()
    return any(phrase in ql for phrase in ("show me the source", "show source", "show sources", "sources"))

def send_question(q):
    if not q:
        return
    st.session_state.history.append({"role": "user", "text": q})
    save_history(st.session_state.history)
    include_sources = wants_sources(q)
    try:
        with st.spinner("Thinking..."):
            resp = requests.post(f"{BACKEND}/query", json={"question": q, "top_k": 4, "include_sources": include_sources}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            ans = data.get("answer", "(no answer)")
            sources = data.get("sources", [])
            st.session_state.history.append({"role": "assistant", "text": ans, "sources": sources})
            save_history(st.session_state.history)
    except Exception as e:
        st.session_state.history.append({"role": "assistant", "text": f"Error: {e}", "sources": []})
        save_history(st.session_state.history)

# Input form
with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask about your activities", "")
    submit = st.form_submit_button("Send")
    if submit:
        send_question(q)

st.divider()

# Controls
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Clear history"):
        st.session_state.history = []
        save_history(st.session_state.history)
with col2:
    if st.button("Reload history"):
        st.session_state.history = load_history()

# Render history (newest first)
for msg in reversed(st.session_state.history):
    if msg["role"] == "assistant":
        st.markdown(f"**Assistant:** {msg['text']}")
        if msg.get("sources"):
            st.markdown(f"_Sources:_ {', '.join(map(str, msg['sources']))}")
        st.write("---")
    else:
        st.markdown(f"**You:** {msg['text']}")
        st.write("")

