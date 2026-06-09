"""
POST /generate — integration endpoint consumed by Person C's Streamlit UI.

Accepts the three-field payload the UI sends, runs RAG retrieval against
Person A's knowledge base, and returns a structured report in the exact
shape the UI expects. No LLM required — the retrieved regulation chunks
populate the citations directly.

Response shape (matches src/backend_client.py mock and app.py rendering):
  {
    "title": str,
    "metadata": {"inspector": str, "status": "DRAFT"},
    "sections": [
      {"heading": str, "body": str, "citations": [{"id": int, "source": str, "chunk_id": str}]}
    ],
    "references": [{"id": int, "source": str, "chunk_id": str, "snippet": str}]
  }
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.retriever import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Report Generation"])


class GenerateRequest(BaseModel):
    notes: str = ""
    transcript: str = ""
    ocr_text: str = ""


@router.post("/generate")
def generate(request: GenerateRequest) -> dict:
    combined = " ".join(
        part.strip()
        for part in [request.notes, request.transcript, request.ocr_text]
        if part.strip()
    )

    if not combined:
        raise HTTPException(status_code=422, detail="At least one of notes, transcript, or ocr_text is required.")

    try:
        chunks = retrieve_relevant_chunks(query=combined, top_k=5)
    except (RuntimeError, EnvironmentError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Retrieval failed during /generate")
        raise HTTPException(status_code=500, detail="Retrieval error.") from exc

    # Build a de-duplicated reference list (one entry per chunk)
    references = [
        {
            "id": i + 1,
            "source": chunk.source.file_name,
            "chunk_id": chunk.chunk_id,
            "snippet": chunk.text[:200].replace("\n", " "),
        }
        for i, chunk in enumerate(chunks)
    ]

    # Map chunk_id → reference id for citation linking
    chunk_to_ref_id = {ref["chunk_id"]: ref["id"] for ref in references}

    # Section 1 — Observations: echo back the engineer's input with all chunk citations
    obs_citations = [
        {"id": ref["id"], "source": ref["source"], "chunk_id": ref["chunk_id"]}
        for ref in references
    ]

    # Section 2 — Applicable Regulations: one sub-section per retrieved chunk
    reg_body_parts = []
    for chunk in chunks:
        src = chunk.source
        section_label = f"§{src.section}" if src.section else src.standard_name
        reg_body_parts.append(
            f"[{section_label} — {src.title}]\n{chunk.text[:400]}"
        )
    reg_body = "\n\n".join(reg_body_parts)
    reg_citations = [
        {"id": chunk_to_ref_id[c.chunk_id], "source": c.source.file_name, "chunk_id": c.chunk_id}
        for c in chunks
    ]

    return {
        "title": "Site Inspection Field Report",
        "metadata": {"inspector": "Field Engineer", "status": "DRAFT"},
        "sections": [
            {
                "heading": "Observations",
                "body": combined,
                "citations": obs_citations,
            },
            {
                "heading": "Applicable Regulations",
                "body": reg_body,
                "citations": reg_citations,
            },
            {
                "heading": "Compliance Assessment",
                "body": (
                    "Based on the retrieved regulatory standards, the observations "
                    "above should be reviewed against the cited clauses. "
                    "Non-conforming items must be documented and escalated per the "
                    "applicable standard's response timeline."
                ),
                "citations": reg_citations[:2],
            },
        ],
        "references": references,
    }
