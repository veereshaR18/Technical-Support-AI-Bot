"""Part B1: semantic retrieval over the local FAISS index."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS

from config import TOP_K


def retrieve_relevant_chunks(vector_store: FAISS, query: str, k: int = TOP_K) -> list[str]:
    """Return the top-k chunks most relevant to a user query."""
    results = vector_store.similarity_search(query, k=k)
    chunks = [doc.page_content for doc in results]

    print(f"[Retriever] Query: {query}")
    for index, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.split())[:120]
        print(f"[Retriever] Source {index}: {preview}...")

    return chunks
