# Upwork API Technical Support Bot

This is a RAG-based Streamlit app that answers developer questions about the
provided Upwork API documentation. It uses local embeddings, a local FAISS vector
store, and a hosted DeepInfra Llama model through an OpenAI-compatible API.

## Stack

- Python
- LangChain
- FAISS
- `sentence-transformers/all-MiniLM-L6-v2` local embeddings
- Streamlit
- DeepInfra OpenAI-compatible chat completions

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add the provided API key:

```bash
DEEPINFRA_API_KEY=your_key_here
```

4. Put the Upwork API PDF at:

```text
docs/API Documentation Partial.pdf
```

## Build the Knowledge Base

```bash
python ingest.py
```

The script prints the required sanity check:

- total character count
- a sample of the loaded text
- number of 500-character chunks with 50-character overlap

## Run the App

```bash
streamlit run app.py
```

The UI displays:

- the generated answer
- the exact retrieved source snippets
- response latency in seconds

## Run the Evaluation Questions

```bash
python evaluate.py
```

## Chunk Overlap Note

The 50-character overlap helps preserve context when a technical example,
endpoint, parameter list, or code snippet lands across a chunk boundary. Without
overlap, the retriever may return only the second half of a curl request or only
the parameter names without the endpoint that explains them.
