"""
End-to-end tests for the ingestion pipeline.

Uses fake embeddings and a tmp_path so the full load → chunk → embed → store
path is exercised without touching OpenAI or the real data/chroma_db directory.
"""
from pathlib import Path

import pytest

from scripts.ingest import run_ingestion

KB_RAW = Path(__file__).resolve().parents[1] / "data" / "kb_raw"


# ---------------------------------------------------------------------------
# Fake embedding model
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    DIM = 8

    def embed_documents(self, texts):
        return [[float(i % 10) / 10] * self.DIM for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.1] * self.DIM


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_ingestion_loads_all_five_documents(tmp_path):
    result = run_ingestion(
        source_dir=KB_RAW,
        embedding_model=_FakeEmbeddings(),
        persist_dir=tmp_path,
    )
    assert result["error"] is None
    assert result["docs_loaded"] == 5


def test_ingestion_creates_chunks(tmp_path):
    result = run_ingestion(
        source_dir=KB_RAW,
        embedding_model=_FakeEmbeddings(),
        persist_dir=tmp_path,
    )
    assert result["chunks_created"] > 5


def test_ingestion_adds_chunks_to_vector_store(tmp_path):
    result = run_ingestion(
        source_dir=KB_RAW,
        embedding_model=_FakeEmbeddings(),
        persist_dir=tmp_path,
    )
    assert result["chunks_added"] > 0
    assert result["stats"]["document_count"] == result["chunks_added"]


def test_ingestion_returns_stats_structure(tmp_path):
    result = run_ingestion(
        source_dir=KB_RAW,
        embedding_model=_FakeEmbeddings(),
        persist_dir=tmp_path,
    )
    assert "collection_name" in result["stats"]
    assert "document_count" in result["stats"]


def test_ingestion_idempotent_on_second_run(tmp_path):
    fake = _FakeEmbeddings()
    result_1 = run_ingestion(
        source_dir=KB_RAW, embedding_model=fake, persist_dir=tmp_path
    )
    result_2 = run_ingestion(
        source_dir=KB_RAW, embedding_model=fake, persist_dir=tmp_path
    )
    assert result_2["chunks_added"] == 0
    assert result_2["stats"]["document_count"] == result_1["chunks_added"]


def test_ingestion_reset_then_reingest(tmp_path):
    fake = _FakeEmbeddings()
    result_1 = run_ingestion(
        source_dir=KB_RAW, embedding_model=fake, persist_dir=tmp_path
    )
    result_2 = run_ingestion(
        source_dir=KB_RAW, embedding_model=fake, persist_dir=tmp_path, reset=True
    )
    assert result_2["chunks_added"] == result_1["chunks_added"]


def test_ingestion_missing_source_dir_returns_error(tmp_path):
    result = run_ingestion(
        source_dir="/tmp/does_not_exist_xyz",
        embedding_model=_FakeEmbeddings(),
        persist_dir=tmp_path,
    )
    assert result["error"] is not None
    assert result["docs_loaded"] == 0
