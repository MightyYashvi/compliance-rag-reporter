#!/usr/bin/env python3
"""
End-to-end smoke test for the RAG pipeline.

Runs ingestion + live retrieval queries and prints a formatted report.
Does NOT require the FastAPI server to be running — it calls the retriever
directly so the full pipeline (load → chunk → embed → store → retrieve)
can be verified in one command.

Usage (from backend/):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --reset              # re-ingest from scratch
    python scripts/smoke_test.py --mock-embeddings --reset   # no API key needed
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest import run_ingestion
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.vector_store import get_vector_store, vector_store_stats

QUERIES = [
    {
        "query": "rust near drainage outlet",
        "filters": {"jurisdiction": "Singapore"},
        "note": "Tests corrosion / drainage semantic match across different word choices",
    },
    {
        "query": "large concrete crack",
        "filters": {"jurisdiction": "Singapore"},
        "note": "Tests structural crack retrieval",
    },
    {
        "query": "water leakage near pedestrian walkway",
        "filters": {"jurisdiction": "Singapore"},
        "note": "Tests water leakage + environmental risk retrieval",
    },
    {
        "query": "unsafe wet walking surface",
        "filters": {"jurisdiction": "Singapore"},
        "note": "Tests site safety hazard retrieval",
    },
    {
        "query": "severe corrosion on load-bearing structure",
        "filters": {"jurisdiction": "Singapore"},
        "note": "Tests critical severity / escalation retrieval",
    },
]


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def run_smoke_test(reset: bool = False, mock_embeddings: bool = False) -> None:
    embedding_model = None
    persist_dir = None
    if mock_embeddings:
        from scripts.dev_utils import MockEmbeddings, MOCK_PERSIST_DIR
        embedding_model = MockEmbeddings()
        persist_dir = MOCK_PERSIST_DIR

    print(_hr("═"))
    print("  Compliance RAG Reporter — End-to-End Smoke Test")
    if mock_embeddings:
        print("  [mock-embeddings mode — no API key required]")
    print(_hr("═"))

    # ── Step 1: Ingest ──────────────────────────────────────────────────────
    print("\n[1/3] Running ingestion pipeline...\n")
    result = run_ingestion(
        reset=reset,
        embedding_model=embedding_model,
        persist_dir=persist_dir,
    )
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        sys.exit(1)

    print(f"  Source documents : {result['docs_loaded']}")
    print(f"  Chunks created   : {result['chunks_created']}")
    print(f"  Chunks added     : {result['chunks_added']}")
    print(f"  Collection total : {result['stats']['document_count']}")

    # ── Step 2: Verify collection ───────────────────────────────────────────
    print("\n[2/3] Verifying vector store...\n")
    vs = get_vector_store(embedding_model=embedding_model, persist_dir=persist_dir)
    stats = vector_store_stats(vs)
    print(f"  Collection : {stats['collection_name']}")
    print(f"  Total docs : {stats['document_count']}")
    if stats["document_count"] == 0:
        print("  ERROR: ChromaDB collection is empty after ingestion.")
        sys.exit(1)
    print("  Vector store OK")

    # ── Step 3: Run retrieval queries ───────────────────────────────────────
    print(f"\n[3/3] Running {len(QUERIES)} retrieval queries...\n")

    for i, q in enumerate(QUERIES, 1):
        print(_hr())
        print(f"Query {i}: \"{q['query']}\"")
        print(f"Note  : {q['note']}")
        print()

        try:
            chunks = retrieve_relevant_chunks(
                query=q["query"],
                top_k=3,
                filters=q.get("filters"),
                vector_store=vs,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        if not chunks:
            print("  No chunks returned.")
            continue

        for j, chunk in enumerate(chunks, 1):
            src = chunk.source
            print(f"  Result {j}  score={chunk.score:.4f}")
            print(f"    Doc    : {src.title}")
            print(f"    File   : {src.file_name}")
            print(f"    Section: {src.section or 'n/a'}")
            print(f"    Std    : {src.standard_name}  |  {src.jurisdiction}")
            wrapped = textwrap.fill(chunk.text[:300], width=68, initial_indent="    > ", subsequent_indent="      ")
            print(wrapped)
            print()

    print(_hr("═"))
    print("  Smoke test complete.")
    print(_hr("═"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG pipeline smoke test.")
    parser.add_argument("--reset", action="store_true", help="Clear and re-ingest collection.")
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help=(
            "Use deterministic fake embeddings — no OPENAI_API_KEY needed. "
            "Reads/writes data/chroma_db_mock, never the real collection."
        ),
    )
    args = parser.parse_args()
    run_smoke_test(reset=args.reset, mock_embeddings=args.mock_embeddings)


if __name__ == "__main__":
    main()
