"""Shared configuration for the Upwork API support bot."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOC_PATH = BASE_DIR / "docs" / "API Documentation Partial.pdf"
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEEPINFRA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
