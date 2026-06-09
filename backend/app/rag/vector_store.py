"""
ChromaDB vector store layer for the Compliance RAG Reporter.

What a vector database is:
    A vector database indexes high-dimensional embedding vectors and supports
    fast approximate-nearest-neighbour (ANN) queries. Given a query vector it
    returns the K stored vectors that are most similar by cosine distance,
    along with their associated text and metadata.

Why ChromaDB:
    ChromaDB runs fully locally with no external service, persists to disk
    between runs, and integrates directly with LangChain. For a student demo
    project with a small regulatory KB this is the right tradeoff — zero
    infrastructure overhead, instant setup, inspectable on-disk files.

How semantic similarity search works:
    1. The query ("corrosion near pipe") is embedded into a vector Q.
    2. ChromaDB computes the cosine similarity between Q and every stored
       chunk vector.
    3. The top-K most similar chunks are returned regardless of exact word
       overlap — so "rust forming near a joint" and "corrosion near drainage
       outlets" both surface for the same query.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.rag.chunker import KBChunk
from app.rag.embedder import embed_chunks, get_embedding_model

logger = logging.getLogger(__name__)

COLLECTION_NAME = "regulatory_kb"

# Default persist path — relative to this file's location inside backend/app/rag/
PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"


# ---------------------------------------------------------------------------
# Vector store initialisation
# ---------------------------------------------------------------------------

def get_vector_store(embedding_model=None, persist_dir: Path | str | None = None):
    """
    Return a LangChain Chroma instance backed by a local persistent collection.

    Both parameters can be overridden in tests:
      embedding_model — pass a fake to avoid OpenAI API calls
      persist_dir     — pass a tmp_path to avoid writing to data/chroma_db
    """
    from langchain_chroma import Chroma

    if embedding_model is None:
        embedding_model = get_embedding_model()
    if persist_dir is None:
        persist_dir = PERSIST_DIR

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(persist_dir),
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def reset_vector_store(persist_dir: Path | str | None = None) -> None:
    """
    Delete the ChromaDB collection so the next ingestion starts from scratch.

    Safe to call even if the collection does not yet exist.
    """
    import chromadb

    if persist_dir is None:
        persist_dir = PERSIST_DIR

    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted collection: %s", COLLECTION_NAME)
    except Exception:
        logger.debug("Collection did not exist — nothing to delete")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def add_chunks_to_vector_store(
    chunks: list[KBChunk],
    vector_store=None,
) -> int:
    """
    Add chunks to the vector store, skipping any whose chunk_id already exists.

    Idempotent: running ingestion twice on the same KB produces exactly the
    same collection state because duplicate chunk_ids are silently skipped.
    This relies on chunk_ids being deterministic (document_id + index + text),
    which the chunker guarantees.

    Returns the number of chunks actually added (0 if all were already present).
    """
    if not chunks:
        return 0

    if vector_store is None:
        vector_store = get_vector_store()

    # Fetch only IDs — avoids pulling vectors or text back across the wire
    existing_ids: set[str] = set(vector_store.get(include=[])["ids"])

    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("All %d chunks already present — nothing to add", len(chunks))
        return 0

    prepared = embed_chunks(new_chunks)

    try:
        from langchain_core.documents import Document as LCDocument
    except ImportError:
        from langchain.schema import Document as LCDocument  # type: ignore[no-redef]

    lc_docs = [
        LCDocument(page_content=text, metadata=meta)
        for text, meta in zip(prepared["texts"], prepared["metadatas"])
    ]

    vector_store.add_documents(lc_docs, ids=prepared["ids"])
    logger.info("Added %d new chunks to '%s'", len(new_chunks), COLLECTION_NAME)
    return len(new_chunks)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def vector_store_stats(vector_store=None) -> dict[str, Any]:
    """Return collection name and total stored chunk count."""
    if vector_store is None:
        vector_store = get_vector_store()

    return {
        "collection_name": COLLECTION_NAME,
        "document_count": vector_store._collection.count(),
    }
