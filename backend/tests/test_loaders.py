import csv
import tempfile
from pathlib import Path

import pytest

from app.rag.loaders import (
    KBDocument,
    load_kb_documents,
    summarize_loaded_documents,
    _make_document_id,
)

KB_RAW = Path(__file__).resolve().parents[1] / "data" / "kb_raw"

REQUIRED_METADATA_KEYS = {
    "document_id",
    "title",
    "file_name",
    "file_path",
    "doc_type",
    "jurisdiction",
    "standard_name",
    "version",
    "page",
    "source_type",
}

EXPECTED_FILES = {
    "singapore_concrete_crack_regulation.txt",
    "singapore_drainage_inspection_standard.txt",
    "singapore_site_safety_guideline.txt",
    "singapore_structural_corrosion_manual.txt",
    "singapore_water_leakage_protocol.txt",
}


def test_loads_all_five_sample_txt_files():
    docs = load_kb_documents(KB_RAW)
    loaded = {doc.metadata["file_name"] for doc in docs}
    assert EXPECTED_FILES.issubset(loaded)


def test_every_document_has_required_metadata_keys():
    docs = load_kb_documents(KB_RAW)
    assert docs, "No documents loaded — check KB_RAW path"
    for doc in docs:
        missing = REQUIRED_METADATA_KEYS - set(doc.metadata.keys())
        assert not missing, f"{doc.metadata['file_name']} is missing keys: {missing}"


def test_document_id_is_deterministic():
    docs_a = load_kb_documents(KB_RAW)
    docs_b = load_kb_documents(KB_RAW)
    ids_a = {doc.metadata["document_id"] for doc in docs_a}
    ids_b = {doc.metadata["document_id"] for doc in docs_b}
    assert ids_a == ids_b


def test_document_id_helper_is_stable():
    assert _make_document_id("foo.txt") == _make_document_id("foo.txt")
    assert _make_document_id("foo.txt") != _make_document_id("bar.txt")


def test_jurisdiction_inferred_as_singapore():
    docs = load_kb_documents(KB_RAW)
    for doc in docs:
        assert doc.metadata["jurisdiction"] == "Singapore", (
            f"{doc.metadata['file_name']} has unexpected jurisdiction"
        )


def test_title_is_non_empty():
    docs = load_kb_documents(KB_RAW)
    for doc in docs:
        assert doc.metadata["title"].strip(), f"{doc.metadata['file_name']} has empty title"


def test_unsupported_files_are_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "valid.txt").write_text("Regulation Title\nSome content.")
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x01\x02")
        (tmp_path / "report.xlsx").write_bytes(b"fake excel data")
        (tmp_path / "archive.zip").write_bytes(b"PK")

        docs = load_kb_documents(tmp_path)
        loaded_names = {doc.metadata["file_name"] for doc in docs}

        assert "valid.txt" in loaded_names
        assert ".DS_Store" not in loaded_names
        assert "report.xlsx" not in loaded_names
        assert "archive.zip" not in loaded_names


def test_missing_directory_returns_empty_list():
    docs = load_kb_documents("/tmp/this_path_definitely_does_not_exist_xyz")
    assert docs == []


def test_metadata_csv_overrides_are_applied():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "doc.txt").write_text("Draft Title\nContent here.")

        csv_path = tmp_path / "metadata.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file_name", "title", "jurisdiction", "doc_type", "standard_name", "version"]
            )
            writer.writeheader()
            writer.writerow({
                "file_name": "doc.txt",
                "title": "Overridden Title",
                "jurisdiction": "Malaysia",
                "doc_type": "guideline",
                "standard_name": "OVR",
                "version": "99",
            })

        docs = load_kb_documents(tmp_path)
        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["title"] == "Overridden Title"
        assert meta["jurisdiction"] == "Malaysia"
        assert meta["doc_type"] == "guideline"
        assert meta["standard_name"] == "OVR"
        assert meta["version"] == "99"


def test_summarize_loaded_documents_structure():
    docs = load_kb_documents(KB_RAW)
    summary = summarize_loaded_documents(docs)

    assert "document_count" in summary
    assert "files" in summary
    assert summary["document_count"] == len(docs)
    assert len(summary["files"]) == len(docs)

    for entry in summary["files"]:
        assert "file_name" in entry
        assert "title" in entry
        assert "jurisdiction" in entry
