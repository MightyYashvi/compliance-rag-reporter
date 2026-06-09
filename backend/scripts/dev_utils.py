"""
Development utilities for running the RAG pipeline without an OpenAI API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# MockEmbeddings is defined in app/rag/embedder — single source of truth.
from app.rag.embedder import MockEmbeddings  # noqa: F401  (re-exported for scripts)

# Separate persist dir so mock runs never pollute the real ChromaDB collection.
MOCK_PERSIST_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db_mock"
