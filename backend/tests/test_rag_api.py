"""
Tests for the POST /rag/retrieve HTTP endpoint.

Uses FastAPI's TestClient and patches retrieve_relevant_chunks so no
real ChromaDB or OpenAI calls are made in the HTTP layer tests.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.schemas import ChunkSource, RetrievedChunk

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id="c1", score=0.87, section="2.2") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Visible corrosion near drainage outlets must be documented.",
        score=score,
        source=ChunkSource(
            document_id="doc001",
            title="Singapore Drainage Inspection Standard (SDIS)",
            file_name="singapore_drainage_inspection_standard.txt",
            page=1,
            section=section,
            doc_type="standard",
            jurisdiction="Singapore",
            standard_name="SDIS",
        ),
    )


_MOCK_TARGET = "app.api.rag_router.retrieve_relevant_chunks"


# ---------------------------------------------------------------------------
# Health check (sanity)
# ---------------------------------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /rag/retrieve — happy path
# ---------------------------------------------------------------------------

def test_retrieve_returns_200():
    with patch(_MOCK_TARGET, return_value=[_make_chunk()]):
        response = client.post(
            "/rag/retrieve",
            json={"query": "Corrosion observed near drainage outlet"},
        )
    assert response.status_code == 200


def test_retrieve_response_contains_query():
    with patch(_MOCK_TARGET, return_value=[_make_chunk()]):
        response = client.post(
            "/rag/retrieve",
            json={"query": "Corrosion observed near drainage outlet"},
        )
    assert response.json()["query"] == "Corrosion observed near drainage outlet"


def test_retrieve_response_chunks_shape():
    with patch(_MOCK_TARGET, return_value=[_make_chunk()]):
        response = client.post(
            "/rag/retrieve",
            json={"query": "Corrosion near outlet"},
        )
    chunk = response.json()["chunks"][0]
    assert "chunk_id" in chunk
    assert "text" in chunk
    assert "score" in chunk
    assert "source" in chunk


def test_retrieve_source_fields_present():
    with patch(_MOCK_TARGET, return_value=[_make_chunk()]):
        response = client.post("/rag/retrieve", json={"query": "drainage"})
    source = response.json()["chunks"][0]["source"]
    for key in ("document_id", "title", "file_name", "page", "section",
                "doc_type", "jurisdiction", "standard_name"):
        assert key in source, f"source missing key: {key}"


def test_retrieve_with_filters():
    with patch(_MOCK_TARGET, return_value=[_make_chunk()]) as mock_fn:
        client.post(
            "/rag/retrieve",
            json={
                "query": "corrosion",
                "top_k": 3,
                "filters": {"jurisdiction": "Singapore", "doc_type": "standard"},
            },
        )
    call_kwargs = mock_fn.call_args
    assert call_kwargs.kwargs["filters"]["jurisdiction"] == "Singapore"
    assert call_kwargs.kwargs["top_k"] == 3


def test_retrieve_empty_chunks_list():
    with patch(_MOCK_TARGET, return_value=[]):
        response = client.post("/rag/retrieve", json={"query": "irrelevant query"})
    assert response.status_code == 200
    assert response.json()["chunks"] == []


def test_retrieve_multiple_chunks():
    chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
    with patch(_MOCK_TARGET, return_value=chunks):
        response = client.post("/rag/retrieve", json={"query": "test"})
    assert len(response.json()["chunks"]) == 3


def test_score_is_float(populated=None):
    with patch(_MOCK_TARGET, return_value=[_make_chunk(score=0.91)]):
        response = client.post("/rag/retrieve", json={"query": "test"})
    assert isinstance(response.json()["chunks"][0]["score"], float)


# ---------------------------------------------------------------------------
# Validation errors (422)
# ---------------------------------------------------------------------------

def test_empty_query_returns_422():
    response = client.post("/rag/retrieve", json={"query": ""})
    assert response.status_code == 422


def test_whitespace_query_returns_422():
    response = client.post("/rag/retrieve", json={"query": "   "})
    assert response.status_code == 422


def test_missing_query_returns_422():
    response = client.post("/rag/retrieve", json={"top_k": 3})
    assert response.status_code == 422


def test_top_k_zero_returns_422():
    response = client.post("/rag/retrieve", json={"query": "test", "top_k": 0})
    assert response.status_code == 422


def test_top_k_above_max_returns_422():
    response = client.post("/rag/retrieve", json={"query": "test", "top_k": 99})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Service errors (503)
# ---------------------------------------------------------------------------

def test_empty_vector_store_returns_503():
    with patch(_MOCK_TARGET, side_effect=RuntimeError("not ingested")):
        response = client.post("/rag/retrieve", json={"query": "test"})
    assert response.status_code == 503


def test_missing_api_key_returns_503():
    with patch(_MOCK_TARGET, side_effect=EnvironmentError("OPENAI_API_KEY not set")):
        response = client.post("/rag/retrieve", json={"query": "test"})
    assert response.status_code == 503
