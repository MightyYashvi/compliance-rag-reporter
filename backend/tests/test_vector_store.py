"""
Tests for the ChromaDB vector store layer.

All tests use a fake embedding model and a pytest tmp_path so no real
OpenAI API calls are made and nothing is written to data/chroma_db/.
"""
import pytest

from app.rag.chunker import KBChunk
from app.rag.vector_store import (
    COLLECTION_NAME,
    add_chunks_to_vector_store,
    get_vector_store,
    reset_vector_store,
    vector_store_stats,
)


# ---------------------------------------------------------------------------
# Fake embedding model — deterministic, no API call
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    """Returns fixed-length vectors so ChromaDB accepts them without OpenAI."""

    DIM = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Each document gets a unique-ish vector based on its first character
        return [[float(ord(t[0]) if t else 0) / 255] * self.DIM for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(ord(text[0]) if text else 0) / 255] * self.DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vs(tmp_path):
    """Fresh vector store in a temp directory with fake embeddings."""
    return get_vector_store(embedding_model=_FakeEmbeddings(), persist_dir=tmp_path)


def _make_chunk(chunk_id: str, text: str = "Some regulation text.") -> KBChunk:
    return KBChunk(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc001",
            "title": "Singapore Test Standard",
            "file_name": "test.txt",
            "file_path": "/kb_raw/test.txt",
            "doc_type": "standard",
            "jurisdiction": "Singapore",
            "standard_name": "STS",
            "version": "2025",
            "page": 1,
            "source_type": "txt",
            "chunk_index": 0,
            "section": "2.1",
            "section_title": "Inspection",
        },
    )


# ---------------------------------------------------------------------------
# get_vector_store
# ---------------------------------------------------------------------------

def test_get_vector_store_returns_instance(tmp_path):
    from langchain_chroma import Chroma
    store = get_vector_store(embedding_model=_FakeEmbeddings(), persist_dir=tmp_path)
    assert isinstance(store, Chroma)


# ---------------------------------------------------------------------------
# add_chunks_to_vector_store
# ---------------------------------------------------------------------------

def test_add_chunks_returns_count(vs):
    chunks = [_make_chunk("c1", "Text A."), _make_chunk("c2", "Text B.")]
    added = add_chunks_to_vector_store(chunks, vector_store=vs)
    assert added == 2


def test_add_empty_chunk_list_returns_zero(vs):
    assert add_chunks_to_vector_store([], vector_store=vs) == 0


def test_chunks_are_retrievable_after_add(vs):
    chunks = [_make_chunk("c1", "Corrosion near the drain.")]
    add_chunks_to_vector_store(chunks, vector_store=vs)
    results = vs.get(ids=["c1"])
    assert "c1" in results["ids"]


def test_metadata_survives_ingestion(vs):
    chunk = _make_chunk("c1")
    add_chunks_to_vector_store([chunk], vector_store=vs)
    result = vs.get(ids=["c1"], include=["metadatas"])
    meta = result["metadatas"][0]
    assert meta["document_id"] == "doc001"
    assert meta["jurisdiction"] == "Singapore"
    assert meta["standard_name"] == "STS"
    assert meta["title"] == "Singapore Test Standard"


def test_duplicate_chunk_ids_not_added_twice(vs):
    chunk = _make_chunk("dup1", "Some text.")
    add_chunks_to_vector_store([chunk], vector_store=vs)
    added_second = add_chunks_to_vector_store([chunk], vector_store=vs)
    assert added_second == 0
    stats = vector_store_stats(vector_store=vs)
    assert stats["document_count"] == 1


def test_second_ingestion_only_adds_new_chunks(vs):
    chunk_a = _make_chunk("a1", "Text A.")
    chunk_b = _make_chunk("b1", "Text B.")
    add_chunks_to_vector_store([chunk_a], vector_store=vs)
    added = add_chunks_to_vector_store([chunk_a, chunk_b], vector_store=vs)
    assert added == 1
    assert vector_store_stats(vector_store=vs)["document_count"] == 2


# ---------------------------------------------------------------------------
# reset_vector_store
# ---------------------------------------------------------------------------

def test_reset_clears_collection(tmp_path):
    vs_before = get_vector_store(embedding_model=_FakeEmbeddings(), persist_dir=tmp_path)
    add_chunks_to_vector_store([_make_chunk("c1")], vector_store=vs_before)
    assert vector_store_stats(vector_store=vs_before)["document_count"] == 1

    reset_vector_store(persist_dir=tmp_path)

    vs_after = get_vector_store(embedding_model=_FakeEmbeddings(), persist_dir=tmp_path)
    assert vector_store_stats(vector_store=vs_after)["document_count"] == 0


def test_reset_on_empty_store_does_not_raise(tmp_path):
    reset_vector_store(persist_dir=tmp_path)  # collection doesn't exist yet


# ---------------------------------------------------------------------------
# vector_store_stats
# ---------------------------------------------------------------------------

def test_vector_store_stats_structure(vs):
    stats = vector_store_stats(vector_store=vs)
    assert "collection_name" in stats
    assert "document_count" in stats
    assert stats["collection_name"] == COLLECTION_NAME


def test_vector_store_stats_count_increases_after_add(vs):
    assert vector_store_stats(vector_store=vs)["document_count"] == 0
    add_chunks_to_vector_store([_make_chunk("c1"), _make_chunk("c2")], vector_store=vs)
    assert vector_store_stats(vector_store=vs)["document_count"] == 2
