"""
Chunker for the Compliance RAG Reporter knowledge base.

Why chunking matters for retrieval:
    Vector search operates on fixed-length embeddings. An embedding of a full
    regulatory document compresses too much information into one vector and
    returns irrelevant sentences alongside the relevant ones. Splitting into
    smaller, overlapping chunks lets the retriever surface the precise clause
    that matches the engineer's observation rather than an entire standard.

Why overlap matters:
    Regulation text often places a threshold value in one sentence and its
    consequence or action requirement in the next. Without overlap, that pair
    can land in non-adjacent chunks, neither of which embeds the full context.
    A 200-character overlap ensures both halves of such a clause are always
    retrievable together.

Metadata contract:
    Every chunk inherits the full provenance metadata from its parent document
    (document_id, title, jurisdiction, …) and adds chunk-specific fields
    (chunk_id, chunk_index, section, section_title). Downstream layers —
    the vector store, the retriever, and the citation formatter — rely on
    this metadata being present and stable.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from app.rag.loaders import KBDocument

logger = logging.getLogger(__name__)

# Matches headings like "Section 2.2 — Corrosion Reporting"
# Handles em dash (—), en dash (–), and ASCII hyphen (-).
_SECTION_RE = re.compile(
    r"Section\s+([\d.]+)\s*[-—–]+\s*(.+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Chunk container
# ---------------------------------------------------------------------------

class KBChunk:
    """A chunk of a KB document with full provenance + chunk-level metadata."""

    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"KBChunk(chunk_id={self.metadata.get('chunk_id')!r}, "
            f"doc={self.metadata.get('file_name')!r}, "
            f"idx={self.metadata.get('chunk_index')}, "
            f"chars={len(self.page_content)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """
    Deterministic 16-char ID stable across repeated ingestion runs.

    Encodes document_id + index + text so that the same chunk always gets the
    same ID — ChromaDB upserts use this to avoid duplicating chunks on
    re-ingestion without a full collection wipe.
    """
    raw = f"{document_id}:{chunk_index}:{text}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _detect_section(
    chunk_text: str, full_doc_text: str
) -> tuple[str | None, str | None]:
    """
    Return (section_number, section_title) for this chunk.

    Strategy:
      1. Look for a section heading inside the chunk itself.
      2. If absent, anchor the chunk's position in the full document using its
         first 80 characters, then walk backwards to find the most recent
         preceding section heading.

    This means a chunk that contains only body text still gets the correct
    section attribution for citation purposes.
    """
    # 1. Heading inside the chunk
    m = _SECTION_RE.search(chunk_text)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # 2. Walk back from where this chunk starts in the full document
    anchor = chunk_text[:80]
    pos = full_doc_text.find(anchor)
    if pos == -1:
        return None, None

    matches = list(_SECTION_RE.finditer(full_doc_text[:pos]))
    if matches:
        last = matches[-1]
        return last.group(1).strip(), last.group(2).strip()

    return None, None


def _get_splitter(chunk_size: int, chunk_overlap: int):
    """Return LangChain RecursiveCharacterTextSplitter or a simple fallback."""
    for import_path in (
        "langchain.text_splitter",
        "langchain_text_splitters",
    ):
        try:
            import importlib
            mod = importlib.import_module(import_path)
            cls = getattr(mod, "RecursiveCharacterTextSplitter")
            return cls(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
        except (ImportError, AttributeError):
            continue

    logger.warning(
        "LangChain text splitter not found — using built-in fallback. "
        "Install langchain or langchain-text-splitters for better splitting."
    )
    return _FallbackSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class _FallbackSplitter:
    """
    Simple character-based splitter used when LangChain is unavailable.
    Splits on character count with a fixed overlap step.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += step
        return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[KBDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[KBChunk]:
    """
    Split loaded KB documents into overlapping chunks ready for vector indexing.

    For each document:
      1. Split raw text using RecursiveCharacterTextSplitter (or fallback).
      2. For each text segment, detect its enclosing section heading.
      3. Build a KBChunk that merges parent metadata with chunk-level fields.

    chunk_index resets to 0 for each new document, so ordering is always
    relative to the source document rather than the full batch.
    """
    splitter = _get_splitter(chunk_size, chunk_overlap)
    all_chunks: list[KBChunk] = []

    for doc in documents:
        raw_chunks = splitter.split_text(doc.page_content)

        for idx, text in enumerate(raw_chunks):
            if not text.strip():
                continue

            section, section_title = _detect_section(text, doc.page_content)

            chunk_meta = {
                **doc.metadata,
                "chunk_id": _make_chunk_id(doc.metadata["document_id"], idx, text),
                "chunk_index": idx,
                "section": section,
                "section_title": section_title,
            }
            all_chunks.append(KBChunk(page_content=text, metadata=chunk_meta))

        logger.info(
            "Chunked %s → %d chunks", doc.metadata.get("file_name"), len(raw_chunks)
        )

    return all_chunks


def summarize_chunks(chunks: list[KBChunk]) -> dict[str, Any]:
    """Return a lightweight summary for logging and smoke-testing."""
    per_doc: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata.get("document_id", "unknown")
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1

    return {
        "chunk_count": len(chunks),
        "documents": per_doc,
    }
