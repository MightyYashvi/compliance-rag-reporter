"""
Retrieval layer for the Compliance RAG Reporter.

This module is the bridge between a raw field observation query and the
ranked regulation chunks that the report generator uses to produce cited
draft reports. It owns three responsibilities:

  1. Embed the query with the same model used during ingestion so the
     similarity search is in the same vector space.

  2. Query ChromaDB for the top-K most semantically similar chunks,
     optionally restricting by metadata filters (jurisdiction, doc_type).

  3. Shape each result into a RetrievedChunk that carries both the
     regulation text and its full citation provenance — no secondary
     lookup required downstream.
"""
from __future__ import annotations

import logging
from typing import Any

from app.rag.schemas import ChunkSource, RetrievedChunk
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Required metadata keys for a chunk to be citable.
_REQUIRED_CITATION_KEYS = {"chunk_id", "document_id", "title", "file_name", "jurisdiction"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_chroma_filter(filters: dict[str, Any] | None) -> dict | None:
    """
    Convert the API filter dict into a ChromaDB `where` clause.

    Skips any filter whose value is None or empty string so callers don't
    need to strip optional fields themselves.
    """
    if not filters:
        return None

    active = {k: v for k, v in filters.items() if v and str(v).strip()}
    if not active:
        return None

    if len(active) == 1:
        key, val = next(iter(active.items()))
        return {key: val}

    return {"$and": [{k: v} for k, v in active.items()]}


def _score_from_distance(raw_distance: float) -> float:
    """
    Convert a ChromaDB L2/cosine distance to a [0, 1] relevance score.

    Uses 1 / (1 + d) so any non-negative distance maps to (0, 1] and
    lower distances always produce higher scores.
    """
    return round(1.0 / (1.0 + raw_distance), 4)


def _to_retrieved_chunk(doc, raw_score: float) -> RetrievedChunk | None:
    """
    Convert a LangChain Document + raw distance into a RetrievedChunk.

    Returns None if the document is missing any required citation field so
    incomplete chunks never reach the API response.
    """
    meta = doc.metadata or {}

    missing = _REQUIRED_CITATION_KEYS - set(meta.keys())
    if missing:
        logger.warning("Dropping chunk missing citation keys: %s", missing)
        return None

    # ChromaDB stores None as "" — restore None for optional fields
    def _or_none(val: str) -> str | None:
        return val if val else None

    source = ChunkSource(
        document_id=meta["document_id"],
        title=meta["title"],
        file_name=meta["file_name"],
        page=int(meta.get("page", 1)),
        section=_or_none(meta.get("section", "")),
        doc_type=meta.get("doc_type", "standard"),
        jurisdiction=meta["jurisdiction"],
        standard_name=meta.get("standard_name", ""),
    )

    return RetrievedChunk(
        chunk_id=meta["chunk_id"],
        text=doc.page_content,
        score=_score_from_distance(raw_score),
        source=source,
    )


def _post_filter(
    results: list[tuple], filters: dict[str, Any]
) -> list[tuple]:
    """Apply metadata filters in Python after retrieval (fallback path)."""
    filtered = []
    for doc, score in results:
        meta = doc.metadata or {}
        if all(
            not v or meta.get(k) == v
            for k, v in filters.items()
            if v and str(v).strip()
        ):
            filtered.append((doc, score))
    return filtered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    vector_store=None,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-K regulation chunks most semantically relevant to query.

    Parameters
    ----------
    query       : field observation text, e.g. "Corrosion near drainage outlet"
    top_k       : number of chunks to return
    filters     : optional dict with doc_type / jurisdiction / standard_name
    vector_store: injected for testing; defaults to production ChromaDB

    Raises
    ------
    ValueError       if query is blank
    RuntimeError     if the vector store is empty (ingestion not run)
    EnvironmentError if OPENAI_API_KEY is missing (propagated from embedder)
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if vector_store is None:
        vector_store = get_vector_store()

    if vector_store._collection.count() == 0:
        raise RuntimeError(
            "The regulatory knowledge base has not been ingested. "
            "Run `python scripts/ingest.py` from the backend/ directory first."
        )

    chroma_filter = _build_chroma_filter(filters)

    try:
        raw_results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=chroma_filter,
        )
    except Exception as exc:
        # ChromaDB may reject a filter if that metadata key never appeared in
        # any document. Fall back to an unfiltered search and post-filter.
        logger.warning(
            "ChromaDB filtered search failed (%s) — falling back to post-filter", exc
        )
        raw_results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k * 3,
        )
        if filters:
            raw_results = _post_filter(raw_results, filters)
        raw_results = raw_results[:top_k]

    chunks: list[RetrievedChunk] = []
    for doc, raw_score in raw_results:
        chunk = _to_retrieved_chunk(doc, raw_score)
        if chunk is not None:
            chunks.append(chunk)

    logger.info(
        "Retrieved %d chunks for query %r (filter=%s)", len(chunks), query[:60], filters
    )
    return chunks
