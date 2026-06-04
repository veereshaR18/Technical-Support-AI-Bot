"""CLI evaluator for the three required ground-truth questions."""

from __future__ import annotations

import time

from dotenv import load_dotenv

from ingest import load_vector_store
from llm import generate_answer
from retriever import retrieve_relevant_chunks


QUESTIONS = [
    "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract details?",
]


def main() -> None:
    load_dotenv()
    vector_store = load_vector_store()

    for question in QUESTIONS:
        print("=" * 80)
        print(f"Question: {question}")
        started_at = time.perf_counter()
        chunks = retrieve_relevant_chunks(vector_store, question, k=3)
        answer = generate_answer(question, chunks)
        latency = time.perf_counter() - started_at
        print(f"Answer: {answer}")
        print(f"Latency: {latency:.2f} seconds")
        print("Sources:")
        for index, chunk in enumerate(chunks, start=1):
            print(f"[{index}] {' '.join(chunk.split())[:500]}")


if __name__ == "__main__":
    main()
