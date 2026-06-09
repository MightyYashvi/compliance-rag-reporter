"""
FastAPI router for POST /rag/retrieve.

This is Person A's integration endpoint. Person B's report generation
backend POSTs a field observation query here and receives ranked regulation
chunks with full citation metadata to ground the draft report.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.rag.retriever import retrieve_relevant_chunks
from app.rag.schemas import RetrievalRequest, RetrievalResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve relevant regulation chunks for a field observation",
)
def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """
    Given a natural-language field observation, return the most semantically
    similar regulation chunks from the Singapore regulatory knowledge base.

    Each returned chunk carries `source` metadata (title, section, jurisdiction)
    so the report generator can produce grounded citations without any further
    lookups.

    **Errors**
    - `422` — query is blank or top_k is out of range
    - `503` — vector store is empty (run ingestion first) or API key missing
    """
    filters = request.filters.model_dump() if request.filters else None

    try:
        chunks = retrieve_relevant_chunks(
            query=request.query,
            top_k=request.top_k,
            filters=filters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, EnvironmentError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during retrieval")
        raise HTTPException(status_code=500, detail="Internal retrieval error.") from exc

    return RetrievalResponse(query=request.query, chunks=chunks)
