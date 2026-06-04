"""Streamlit interface for the Upwork API Technical Support AI Bot."""

from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_DOC_PATH, TOP_K
from ingest import build_vector_store, load_and_chunk_documents, load_vector_store
from llm import generate_answer
from retriever import retrieve_relevant_chunks


load_dotenv()

st.set_page_config(
    page_title="Upwork API Support Bot",
    layout="wide",
)

st.markdown(
    """
    <style>
    .source-box {
        border: 1px solid #d6dde8;
        border-radius: 8px;
        padding: 0.75rem;
        background: #f8fafc;
                color: #0f172a;
margin-bottom: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .source-box strong {
        color: #1e293b;
    }
    .latency {
        color: #475569;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_or_load_vector_store():
    """Load the cached FAISS index once per Streamlit session."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()
    return st.session_state.vector_store


with st.sidebar:
    st.header("Knowledge Base")
    doc_path = st.text_input("Documentation path", value=str(DEFAULT_DOC_PATH))

    if st.button("Build / Rebuild Index", use_container_width=True):
        try:
            with st.spinner("Reading, chunking, and embedding documentation..."):
                chunks = load_and_chunk_documents(Path(doc_path))
                st.session_state.vector_store = build_vector_store(chunks)
            st.success(f"Indexed {len(chunks)} chunks.")
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.caption("Evaluation questions")
    eval_questions = [
        "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
        "How long is an OAuth access token valid for?",
        "Can I use a Client Credentials Grant to access a user's private contract details?",
    ]
    for index, question in enumerate(eval_questions, start=1):
        if st.button(f"Q{index}", use_container_width=True):
            st.session_state.pending_question = question


st.title("Upwork API Technical Support Bot")
st.caption("RAG with local embeddings, FAISS retrieval, DeepInfra Llama, and source snippets.")

pending_question = st.session_state.pop("pending_question", "")
query = st.text_area(
    "Ask a developer support question",
    value=pending_question,
    placeholder="Example: How long is an OAuth access token valid for?",
    height=100,
)

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        try:
            vector_store = get_or_load_vector_store()
            started_at = time.perf_counter()
            chunks = retrieve_relevant_chunks(vector_store, query.strip(), k=TOP_K)
            answer = generate_answer(query.strip(), chunks)
            latency = time.perf_counter() - started_at

            st.subheader("Answer")
            st.write(answer)
            st.markdown(f'<div class="latency">Latency: {latency:.2f} seconds</div>', unsafe_allow_html=True)

            st.subheader("Sources")
            for index, chunk in enumerate(chunks, start=1):
                safe_chunk = html.escape(chunk)
                st.markdown(
                    f'<div class="source-box"><strong>Source {index}</strong><br>{safe_chunk}</div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(str(exc))


