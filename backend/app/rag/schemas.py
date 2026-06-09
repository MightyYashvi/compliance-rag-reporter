"""
Pydantic request/response schemas for the POST /rag/retrieve endpoint.

These models are the integration contract between Person A's retrieval layer
and Person B's report generation backend. Any field change here must be
coordinated with Person B.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RetrievalFilters(BaseModel):
    """Optional metadata filters applied during or after vector search."""

    doc_type: str | None = None
    jurisdiction: str | None = None
    standard_name: str | None = None


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Field observation text to retrieve regulations for.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to return.")
    filters: RetrievalFilters | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class ChunkSource(BaseModel):
    """Full provenance for a single retrieved chunk — used for citations."""

    document_id: str
    title: str
    file_name: str
    page: int
    section: str | None
    doc_type: str
    jurisdiction: str
    standard_name: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = Field(..., description="Relevance score in [0, 1]; higher = more relevant.")
    source: ChunkSource


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
