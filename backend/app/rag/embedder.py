"""
Embedding layer for the Compliance RAG Reporter.

What embeddings are:
    An embedding converts a piece of text into a dense numerical vector —
    a list of floating-point numbers that encode the semantic meaning of the
    text. The model is trained so that texts with similar meaning produce
    vectors that are close together in that high-dimensional space, even when
    the two texts share no exact keywords.

Why semantic vectors matter for retrieval:
    A field engineer might write "rust forming around a pipe joint" while the
    Singapore standard says "corrosion near drainage outlets". A keyword
    search misses this. Embedding-based retrieval finds it because both
    phrases map to nearby vectors in the semantic space.

Why metadata must travel with embeddings:
    ChromaDB stores each chunk as a (vector, text, metadata) triple. The
    metadata carries the citation data — document title, standard name,
    section, jurisdiction — that the report generator needs to produce
    compliant source references. Without it a retrieved chunk is an
    unattributed fragment of unknown origin.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.rag.chunker import KBChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

class MockEmbeddings:
    """
    Deterministic fake embedding model for local development and testing.

    Active only when USE_MOCK_EMBEDDINGS=1 is set in the environment.
    Never used in production — the env var is not set in production deployments.
    """

    DIM = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(ord(t[0]) if t else 0) / 255] * self.DIM for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(ord(text[0]) if text else 0) / 255] * self.DIM


def get_embedding_model():
    """
    Return an embedding model.

    If USE_MOCK_EMBEDDINGS=1 is set, returns MockEmbeddings (no API key needed).
    Otherwise returns the real OpenAIEmbeddings and raises EnvironmentError if
    OPENAI_API_KEY is absent.
    """
    if os.getenv("USE_MOCK_EMBEDDINGS") == "1":
        logger.info("USE_MOCK_EMBEDDINGS=1 — using MockEmbeddings (no API call)")
        return MockEmbeddings()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Copy backend/.env.example to backend/.env and add your key, "
            "then re-run the ingestion script."
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Metadata sanitisation
# ---------------------------------------------------------------------------

def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Convert metadata values to types ChromaDB accepts (str, int, float, bool).

    ChromaDB rejects None and complex types silently or with opaque errors.
    Normalising here keeps the citation fields intact — every field that the
    loader and chunker attached is preserved, just type-coerced.
    """
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif value is None:
            out[key] = ""
        else:
            out[key] = str(value)
    return out


# ---------------------------------------------------------------------------
# Chunk preparation
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[KBChunk]) -> dict[str, list]:
    """
    Prepare KBChunks for insertion into the vector store.

    Returns a dict of parallel lists:
      ids        — deterministic chunk IDs used for deduplication on re-ingestion
      texts      — raw text content that will be embedded by the vector store
      metadatas  — sanitised metadata dicts carrying full citation provenance

    The actual OpenAI API call to produce vectors happens inside the vector
    store (via its embedding_function) when add_documents() is called — not
    here. Separating preparation from the API call keeps tests fast and free
    of network dependencies.
    """
    return {
        "ids": [c.metadata["chunk_id"] for c in chunks],
        "texts": [c.page_content for c in chunks],
        "metadatas": [sanitize_metadata(c.metadata) for c in chunks],
    }
