#!/usr/bin/env python3
"""
One-shot KB ingestion script: load → chunk → embed → store.

Run from backend/:
    python scripts/ingest.py
    python scripts/ingest.py --reset
    python scripts/ingest.py --source-dir data/kb_raw --chunk-size 800 --chunk-overlap 150
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Make `from app.rag.*` work when the script is run directly from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.loaders import load_kb_documents, summarize_loaded_documents
from app.rag.chunker import chunk_documents, summarize_chunks
from app.rag.vector_store import (
    COLLECTION_NAME,
    PERSIST_DIR,
    add_chunks_to_vector_store,
    get_vector_store,
    reset_vector_store,
    vector_store_stats,
)


def run_ingestion(
    source_dir: Path | str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reset: bool = False,
    embedding_model=None,
    persist_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Core ingestion pipeline, separated from CLI argument parsing for testability.

    Returns a summary dict so callers (tests, scripts) can inspect results.
    """
    # Step 1 — Load
    load_kwargs: dict[str, Any] = {}
    if source_dir is not None:
        load_kwargs["source_dir"] = Path(source_dir)

    docs = load_kb_documents(**load_kwargs)
    if not docs:
        return {
            "docs_loaded": 0,
            "chunks_created": 0,
            "chunks_added": 0,
            "stats": {},
            "error": "No documents found. Check source_dir.",
        }

    # Step 2 — Chunk
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3 — Reset (if requested) then store
    effective_persist_dir = persist_dir or PERSIST_DIR
    if reset:
        reset_vector_store(persist_dir=effective_persist_dir)

    vs = get_vector_store(embedding_model=embedding_model, persist_dir=effective_persist_dir)
    added = add_chunks_to_vector_store(chunks, vector_store=vs)
    stats = vector_store_stats(vector_store=vs)

    return {
        "docs_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_added": added,
        "stats": stats,
        "error": None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest regulatory KB documents into ChromaDB."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB collection before ingestion.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        metavar="PATH",
        help="Path to kb_raw directory (default: backend/data/kb_raw).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        metavar="N",
        help="Max characters per chunk (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        metavar="N",
        help="Overlap characters between consecutive chunks (default: 200).",
    )
    return parser.parse_args()


def _print_summary(result: dict[str, Any]) -> None:
    print("\n=== Compliance RAG Reporter — KB Ingestion ===\n")

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return

    stats = result.get("stats", {})
    print(f"  Source documents : {result['docs_loaded']}")
    print(f"  Chunks created   : {result['chunks_created']}")
    print(f"  Chunks added     : {result['chunks_added']}")
    print(f"  Collection total : {stats.get('document_count', '?')}")
    print(f"  Collection name  : {stats.get('collection_name', COLLECTION_NAME)}")
    print(f"  Persist dir      : {PERSIST_DIR}")
    print()

    if result["chunks_added"] == 0:
        print("  All chunks already present — collection is up to date.")
    else:
        print(f"  Ingestion complete. {result['chunks_added']} new chunk(s) stored.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _parse_args()

    result = run_ingestion(
        source_dir=args.source_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )

    _print_summary(result)

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
