# Technical Summary

- I built the bot as a small, explainable RAG pipeline: `ingest.py` loads the supplied Upwork API PDF, prints a sanity check, splits it into 500-character chunks with 50-character overlap, embeds chunks locally with `sentence-transformers/all-MiniLM-L6-v2`, and stores them in FAISS.
- The main challenge was keeping answers grounded. The system prompt makes the model act as a Senior Upwork API Consultant and requires the exact fallback sentence when the retrieved snippets do not contain an answer.
- Chunking technical documentation is tricky because endpoint examples, parameter tables, and curl snippets can cross chunk boundaries. The 50-character overlap reduces the chance that a retrieved chunk loses the context needed to understand a snippet.
- API latency is measured around retrieval plus the DeepInfra chat completion call and displayed in Streamlit, so users can see how long each response took.
- I used GPT/Codex to help structure the project, check the assignment requirements, and draft implementation patterns, but the code is intentionally kept direct so I can explain each function line by line.

## Why I Fit the ProAnalyst AI Team

- I focus on correctness before polish, especially for support bots where hallucinations can mislead developers.
- I can connect product requirements to practical engineering decisions, such as local embeddings for privacy and source snippets for trust.
- I communicate tradeoffs clearly and build in a way that teammates can review, run, and extend without guessing how the system works.
