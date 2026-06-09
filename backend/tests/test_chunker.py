from collections import defaultdict
from pathlib import Path

import pytest

from app.rag.chunker import (
    KBChunk,
    _detect_section,
    _make_chunk_id,
    chunk_documents,
    summarize_chunks,
)
from app.rag.loaders import load_kb_documents

KB_RAW = Path(__file__).resolve().parents[1] / "data" / "kb_raw"

REQUIRED_CHUNK_KEYS = {
    "chunk_id",
    "document_id",
    "title",
    "file_name",
    "file_path",
    "doc_type",
    "jurisdiction",
    "standard_name",
    "chunk_index",
    "source_type",
}


@pytest.fixture(scope="module")
def sample_chunks():
    docs = load_kb_documents(KB_RAW)
    return chunk_documents(docs)


# ---------------------------------------------------------------------------
# Basic creation
# ---------------------------------------------------------------------------

def test_chunks_are_created_from_all_five_documents(sample_chunks):
    source_files = {c.metadata["file_name"] for c in sample_chunks}
    assert len(source_files) == 5


def test_chunk_count_is_greater_than_document_count(sample_chunks):
    # Each ~1 KB document should produce at least 1 chunk with default settings
    assert len(sample_chunks) >= 5


def test_chunk_text_is_non_empty(sample_chunks):
    for chunk in sample_chunks:
        assert chunk.page_content.strip(), f"empty chunk: {chunk}"


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------

def test_every_chunk_has_chunk_id(sample_chunks):
    for chunk in sample_chunks:
        assert chunk.metadata.get("chunk_id"), f"missing chunk_id: {chunk}"


def test_every_chunk_has_required_metadata_keys(sample_chunks):
    for chunk in sample_chunks:
        missing = REQUIRED_CHUNK_KEYS - set(chunk.metadata.keys())
        assert not missing, (
            f"{chunk.metadata.get('file_name')} chunk {chunk.metadata.get('chunk_index')} "
            f"missing keys: {missing}"
        )


def test_every_chunk_preserves_parent_provenance(sample_chunks):
    for chunk in sample_chunks:
        assert chunk.metadata["document_id"]
        assert chunk.metadata["title"]
        assert chunk.metadata["file_name"]
        assert chunk.metadata["jurisdiction"] == "Singapore"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_chunk_id_is_deterministic_across_runs():
    docs = load_kb_documents(KB_RAW)
    ids_a = [c.metadata["chunk_id"] for c in chunk_documents(docs)]
    ids_b = [c.metadata["chunk_id"] for c in chunk_documents(docs)]
    assert ids_a == ids_b


def test_make_chunk_id_is_stable():
    assert _make_chunk_id("doc1", 0, "text") == _make_chunk_id("doc1", 0, "text")


def test_make_chunk_id_differs_for_different_inputs():
    base = _make_chunk_id("doc1", 0, "text")
    assert base != _make_chunk_id("doc1", 1, "text")    # different index
    assert base != _make_chunk_id("doc2", 0, "text")    # different document
    assert base != _make_chunk_id("doc1", 0, "other")   # different text


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_chunk_index_is_sequential_per_document(sample_chunks):
    by_doc: dict[str, list[int]] = defaultdict(list)
    for chunk in sample_chunks:
        by_doc[chunk.metadata["document_id"]].append(chunk.metadata["chunk_index"])

    for doc_id, indices in by_doc.items():
        assert indices == list(range(len(indices))), (
            f"chunk_index not sequential for document {doc_id}: {indices}"
        )


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def test_section_detected_when_heading_is_in_chunk():
    text = "Section 2.2 — Corrosion Reporting\nVisible corrosion near drains."
    section, title = _detect_section(text, text)
    assert section == "2.2"
    assert "Corrosion" in title


def test_section_detected_by_looking_back_in_document():
    full = (
        "Section 3.1 — Photographic Evidence\n"
        "All reports must include photos.\n\n"
        "Photos must show the full extent of damage."
    )
    # Chunk contains only the body text after the heading
    chunk = "Photos must show the full extent of damage."
    section, title = _detect_section(chunk, full)
    assert section == "3.1"
    assert "Photographic" in title


def test_section_number_with_multiple_parts():
    text = "Section 4.0 — Compliance Severity Levels\nMinor: cosmetic issues."
    section, title = _detect_section(text, text)
    assert section == "4.0"


def test_section_returns_none_when_no_headings_present():
    text = "Some generic text with no section headings at all."
    section, title = _detect_section(text, text)
    assert section is None
    assert title is None


def test_real_kb_chunks_have_section_on_at_least_some_chunks(sample_chunks):
    sections_found = [
        c.metadata.get("section")
        for c in sample_chunks
        if c.metadata.get("section") is not None
    ]
    assert len(sections_found) > 0, "No section headings detected in any chunk"


# ---------------------------------------------------------------------------
# summarize_chunks
# ---------------------------------------------------------------------------

def test_summarize_chunks_structure(sample_chunks):
    summary = summarize_chunks(sample_chunks)
    assert "chunk_count" in summary
    assert "documents" in summary
    assert summary["chunk_count"] == len(sample_chunks)
    assert sum(summary["documents"].values()) == len(sample_chunks)


def test_summarize_chunks_has_one_entry_per_source_document(sample_chunks):
    summary = summarize_chunks(sample_chunks)
    assert len(summary["documents"]) == 5
