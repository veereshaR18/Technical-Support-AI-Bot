"""Part A: ingest the Upwork API documentation and build a FAISS index."""

from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_DOC_PATH, EMBEDDING_MODEL, VECTOR_STORE_PATH


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create a local embedding model. No hosted API call is used for embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(file_path: str | Path):
    """Load either the supplied PDF reference or a plain text export."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Documentation file not found: {path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError("Only .pdf, .txt, and .md documentation files are supported.")

    return loader.load()


def chunk_documents(documents) -> list[str]:
    """Split documents into 500-character chunks with 50-character overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content.strip() for chunk in chunks if chunk.page_content.strip()]


def load_and_chunk_documents(file_path: str | Path = DEFAULT_DOC_PATH) -> list[str]:
    """Load docs, print the required sanity check, and return text chunks."""
    documents = load_documents(file_path)
    full_text = "\n".join(doc.page_content for doc in documents)

    print(f"[Sanity Check] Total characters loaded: {len(full_text):,}")
    print("[Sanity Check] Sample:")
    print(full_text[:500])

    chunks = chunk_documents(documents)
    print(
        f"[Chunking] Created {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})."
    )
    return chunks


def build_vector_store(chunks: list[str], save_path: str | Path = VECTOR_STORE_PATH) -> FAISS:
    """Embed chunks locally and persist them to a FAISS vector store."""
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    vector_store = FAISS.from_texts(chunks, embedding=get_embeddings())
    vector_store.save_local(str(save_path))
    print(f"[Vector Store] Saved {len(chunks)} vectors to {save_path}.")
    return vector_store


def load_vector_store(save_path: str | Path = VECTOR_STORE_PATH) -> FAISS:
    """Load the locally persisted FAISS index."""
    path = Path(save_path)
    if not path.exists():
        raise FileNotFoundError(f"Vector store not found at {path}. Run ingest.py first.")

    return FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def main() -> None:
    chunks = load_and_chunk_documents(DEFAULT_DOC_PATH)
    build_vector_store(chunks)


if __name__ == "__main__":
    main()
