"""
Tests for the retriever module.

Uses a real ChromaDB instance (tmp_path) with fake embeddings so no
OpenAI API calls are made. Chunks are pre-populated via the ingestion
pipeline to give the similarity search something to work with.
"""
from pathlib import Path

import pytest

from app.rag.retriever import (
    _build_chroma_filter,
    _score_from_distance,
    retrieve_relevant_chunks,
)
from app.rag.vector_store import add_chunks_to_vector_store, get_vector_store
from app.rag.loaders import load_kb_documents
from app.rag.chunker import chunk_documents

KB_RAW = Path(__file__).resolve().parents[1] / "data" / "kb_raw"


# ---------------------------------------------------------------------------
# Fake embedding model
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    DIM = 8

    def embed_documents(self, texts):
        return [[float(ord(t[0]) if t else 0) / 255] * self.DIM for t in texts]

    def embed_query(self, text):
        return [float(ord(text[0]) if text else 0) / 255] * self.DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def populated_vs(tmp_path_factory):
    """Vector store pre-loaded with all 5 KB documents."""
    tmp = tmp_path_factory.mktemp("chroma")
    fake = _FakeEmbeddings()
    vs = get_vector_store(embedding_model=fake, persist_dir=tmp)
    docs = load_kb_documents(KB_RAW)
    chunks = chunk_documents(docs)
    add_chunks_to_vector_store(chunks, vector_store=vs)
    return vs


@pytest.fixture()
def empty_vs(tmp_path):
    return get_vector_store(embedding_model=_FakeEmbeddings(), persist_dir=tmp_path)


# ---------------------------------------------------------------------------
# _build_chroma_filter
# ---------------------------------------------------------------------------

def test_build_filter_returns_none_when_no_filters():
    assert _build_chroma_filter(None) is None
    assert _build_chroma_filter({}) is None


def test_build_filter_single_key():
    result = _build_chroma_filter({"jurisdiction": "Singapore"})
    assert result == {"jurisdiction": "Singapore"}


def test_build_filter_skips_none_values():
    result = _build_chroma_filter({"jurisdiction": "Singapore", "standard_name": None})
    assert result == {"jurisdiction": "Singapore"}


def test_build_filter_multiple_keys_uses_and():
    result = _build_chroma_filter({"jurisdiction": "Singapore", "doc_type": "standard"})
    assert result is not None
    assert "$and" in result


def test_build_filter_skips_empty_string():
    result = _build_chroma_filter({"jurisdiction": "", "doc_type": "standard"})
    assert result == {"doc_type": "standard"}


# ---------------------------------------------------------------------------
# _score_from_distance
# ---------------------------------------------------------------------------

def test_score_zero_distance_is_one():
    assert _score_from_distance(0.0) == 1.0


def test_score_decreases_as_distance_increases():
    assert _score_from_distance(0.5) > _score_from_distance(1.0)
    assert _score_from_distance(1.0) > _score_from_distance(2.0)


def test_score_is_always_positive():
    for d in [0.0, 0.5, 1.0, 5.0, 100.0]:
        assert _score_from_distance(d) > 0.0


def test_score_is_at_most_one():
    assert _score_from_distance(0.0) <= 1.0


# ---------------------------------------------------------------------------
# retrieve_relevant_chunks — happy path
# ---------------------------------------------------------------------------

def test_retrieval_returns_list(populated_vs):
    results = retrieve_relevant_chunks(
        "Corrosion near drainage outlet", top_k=3, vector_store=populated_vs
    )
    assert isinstance(results, list)


def test_retrieval_respects_top_k(populated_vs):
    results = retrieve_relevant_chunks("inspection", top_k=2, vector_store=populated_vs)
    assert len(results) <= 2


def test_every_result_has_chunk_id(populated_vs):
    results = retrieve_relevant_chunks("corrosion", top_k=3, vector_store=populated_vs)
    for r in results:
        assert r.chunk_id


def test_every_result_has_score_in_range(populated_vs):
    results = retrieve_relevant_chunks("drainage", top_k=3, vector_store=populated_vs)
    for r in results:
        assert 0.0 < r.score <= 1.0


def test_every_result_has_non_empty_text(populated_vs):
    results = retrieve_relevant_chunks("drainage", top_k=3, vector_store=populated_vs)
    for r in results:
        assert r.text.strip()


# ---------------------------------------------------------------------------
# Citation metadata completeness
# ---------------------------------------------------------------------------

def test_every_result_has_complete_source(populated_vs):
    results = retrieve_relevant_chunks("corrosion", top_k=5, vector_store=populated_vs)
    for r in results:
        assert r.source.document_id
        assert r.source.title
        assert r.source.file_name
        assert r.source.jurisdiction == "Singapore"
        assert r.source.doc_type


def test_source_standard_name_is_present(populated_vs):
    results = retrieve_relevant_chunks("crack width", top_k=3, vector_store=populated_vs)
    for r in results:
        assert r.source.standard_name


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_empty_query_raises_value_error(populated_vs):
    with pytest.raises(ValueError, match="empty"):
        retrieve_relevant_chunks("", vector_store=populated_vs)


def test_blank_query_raises_value_error(populated_vs):
    with pytest.raises(ValueError):
        retrieve_relevant_chunks("   ", vector_store=populated_vs)


def test_empty_vector_store_raises_runtime_error(empty_vs):
    with pytest.raises(RuntimeError, match="ingested"):
        retrieve_relevant_chunks("corrosion", vector_store=empty_vs)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_jurisdiction_filter_limits_results(populated_vs):
    results = retrieve_relevant_chunks(
        "inspection",
        top_k=5,
        filters={"jurisdiction": "Singapore"},
        vector_store=populated_vs,
    )
    for r in results:
        assert r.source.jurisdiction == "Singapore"


def test_nonexistent_jurisdiction_returns_empty(populated_vs):
    results = retrieve_relevant_chunks(
        "inspection",
        top_k=5,
        filters={"jurisdiction": "NonexistentCountry"},
        vector_store=populated_vs,
    )
    assert results == []
