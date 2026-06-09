"""Tests for the embedding preparation layer (no real OpenAI calls)."""
import os
from unittest.mock import patch

import pytest

from app.rag.chunker import KBChunk
from app.rag.embedder import embed_chunks, sanitize_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id="abc123", text="Some regulatory text.", extra=None):
    meta = {
        "chunk_id": chunk_id,
        "document_id": "doc001",
        "title": "Singapore Test Standard",
        "file_name": "test.txt",
        "file_path": "/data/kb_raw/test.txt",
        "doc_type": "standard",
        "jurisdiction": "Singapore",
        "standard_name": "STS",
        "version": "2025",
        "page": 1,
        "source_type": "txt",
        "chunk_index": 0,
        "section": "2.1",
        "section_title": "Inspection Requirements",
    }
    if extra:
        meta.update(extra)
    return KBChunk(page_content=text, metadata=meta)


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

def test_embed_chunks_returns_required_keys():
    chunks = [_make_chunk("id1", "Text one."), _make_chunk("id2", "Text two.")]
    result = embed_chunks(chunks)
    assert set(result.keys()) == {"ids", "texts", "metadatas"}


def test_embed_chunks_parallel_lists_same_length():
    chunks = [_make_chunk("id1"), _make_chunk("id2"), _make_chunk("id3")]
    result = embed_chunks(chunks)
    assert len(result["ids"]) == len(result["texts"]) == len(result["metadatas"]) == 3


def test_embed_chunks_ids_match_chunk_ids():
    chunks = [_make_chunk("aaa"), _make_chunk("bbb")]
    result = embed_chunks(chunks)
    assert result["ids"] == ["aaa", "bbb"]


def test_embed_chunks_texts_match_page_content():
    chunks = [_make_chunk("x", "First text."), _make_chunk("y", "Second text.")]
    result = embed_chunks(chunks)
    assert result["texts"] == ["First text.", "Second text."]


def test_embed_chunks_metadata_contains_all_required_fields():
    chunk = _make_chunk()
    result = embed_chunks([chunk])
    meta = result["metadatas"][0]
    for key in ("chunk_id", "document_id", "title", "file_name", "jurisdiction"):
        assert key in meta, f"metadata missing key: {key}"


def test_embed_chunks_empty_list():
    result = embed_chunks([])
    assert result == {"ids": [], "texts": [], "metadatas": []}


# ---------------------------------------------------------------------------
# sanitize_metadata
# ---------------------------------------------------------------------------

def test_sanitize_none_becomes_empty_string():
    result = sanitize_metadata({"section": None, "version": None})
    assert result["section"] == ""
    assert result["version"] == ""


def test_sanitize_primitives_pass_through():
    meta = {"score": 0.87, "page": 3, "active": True, "title": "Reg"}
    result = sanitize_metadata(meta)
    assert result["score"] == 0.87
    assert result["page"] == 3
    assert result["active"] is True
    assert result["title"] == "Reg"


def test_sanitize_non_primitive_becomes_string():
    result = sanitize_metadata({"nested": {"key": "val"}, "lst": [1, 2]})
    assert isinstance(result["nested"], str)
    assert isinstance(result["lst"], str)


def test_sanitize_preserves_all_keys():
    meta = {"a": 1, "b": None, "c": "hello", "d": [1, 2]}
    result = sanitize_metadata(meta)
    assert set(result.keys()) == {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# get_embedding_model — error handling
# ---------------------------------------------------------------------------

def test_get_embedding_model_raises_without_api_key():
    from app.rag.embedder import get_embedding_model
    with patch.dict(os.environ, {}, clear=True):
        # Ensure the key is absent
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            get_embedding_model()
