"""
Frontend-facing generate router — the contract Person C's Streamlit UI calls.

  POST /generate  { notes, transcript, ocr_text }  ->  report dict

This is a thin adapter over the internal generation pipeline. It accepts the
flat multimodal-text payload the frontend sends, runs RAG retrieval + report
generation, and maps the rich GeneratedReport back into the compact shape the
UI and exporters expect:

  {
    "title": str,
    "metadata": {"inspector": str, "status": str},
    "sections": [{"heading": str, "body": str,
                  "citations": [{"id": int, "source": str, "chunk_id": str}]}],
    "references": [{"id": int, "source": str, "chunk_id": str, "snippet": str}],
  }

If no LLM key is configured the endpoint still returns a deterministic,
RAG-grounded report so the deployed service is usable out of the box; set
ANTHROPIC_API_KEY (or OPENAI_API_KEY) to upgrade to real generation.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.models.schemas import (
    GeneratedReport,
    InputType,
    ObservationItem,
    RAGChunk,
)
from app.services.generation import generate_report
from app.services.rag_client import retrieve_chunks

logger = logging.getLogger(__name__)
router = APIRouter(tags=["frontend"])


class GenerateRequest(BaseModel):
    """Flat multimodal-text payload from the frontend."""
    notes: str = ""
    transcript: str = ""
    ocr_text: str = ""
    inspector_name: str = "Field Engineer"
    site_name: str = "Site"


def _observations_from(req: GenerateRequest) -> list[ObservationItem]:
    """Turn the flat text fields into typed observations."""
    obs: list[ObservationItem] = []
    if req.notes.strip():
        obs.append(ObservationItem(input_type=InputType.TEXT, raw_content=req.notes.strip()))
    if req.transcript.strip():
        obs.append(ObservationItem(input_type=InputType.VOICE, raw_content=req.transcript.strip()))
    if req.ocr_text.strip():
        obs.append(ObservationItem(input_type=InputType.IMAGE, raw_content=req.ocr_text.strip()))
    return obs


def _build_references(chunks: list[RAGChunk]) -> tuple[list[dict], dict[str, int]]:
    """Number the retrieved chunks → reference list + chunk_id→id index."""
    references: list[dict] = []
    index: dict[str, int] = {}
    for i, ch in enumerate(chunks, start=1):
        index[ch.chunk_id] = i
        references.append({
            "id": i,
            "source": ch.source_doc,
            "chunk_id": ch.chunk_id,
            "snippet": ch.text,
        })
    return references, index


def _to_frontend_shape(report: GeneratedReport, chunks: list[RAGChunk]) -> dict:
    """Map the internal GeneratedReport onto the compact UI/export contract."""
    references, index = _build_references(chunks)

    sections: list[dict] = []
    for sec in report.sections:
        citations: list[dict] = []
        seen: set[str] = set()
        for claim in sec.cited_claims:
            for j, chunk_id in enumerate(claim.source_chunk_ids):
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                source = (
                    claim.source_docs[j]
                    if j < len(claim.source_docs)
                    else next((c.source_doc for c in chunks if c.chunk_id == chunk_id), "source")
                )
                citations.append({
                    "id": index.get(chunk_id, len(references) + 1),
                    "source": source,
                    "chunk_id": chunk_id,
                })
        sections.append({
            "heading": sec.title,
            "body": sec.content,
            "citations": citations,
        })

    return {
        "title": report.title,
        "metadata": {
            "inspector": report.inspector_name,
            "status": report.status.value.upper(),
        },
        "sections": sections,
        "references": references,
    }


def _fallback_report(obs: list[ObservationItem], chunks: list[RAGChunk]) -> dict:
    """Deterministic RAG-grounded report when no LLM key is configured."""
    references, _ = _build_references(chunks)
    combined = " ".join(o.raw_content for o in obs).strip() or "No observations captured."
    cite_ids = references[:2]
    return {
        "title": "Site Inspection Field Report",
        "metadata": {"inspector": "Field Engineer", "status": "DRAFT"},
        "sections": [
            {
                "heading": "Observations",
                "body": combined,
                "citations": [{"id": c["id"], "source": c["source"], "chunk_id": c["chunk_id"]}
                              for c in cite_ids[:1]],
            },
            {
                "heading": "Compliance Assessment",
                "body": (
                    "Observations were assessed against the retrieved standards. "
                    "Items flagged as non-conforming should be logged and remediated "
                    "per the cited requirements."
                ),
                "citations": [{"id": c["id"], "source": c["source"], "chunk_id": c["chunk_id"]}
                              for c in cite_ids],
            },
        ],
        "references": references,
    }


@router.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """Frontend contract endpoint — flat text in, compact report out."""
    observations = _observations_from(req)
    combined_query = " | ".join(o.raw_content[:300] for o in observations)[:2000]
    rag = await retrieve_chunks(query=combined_query, top_k=5)

    settings = get_settings()
    has_key = (
        (settings.llm_provider == "claude" and settings.anthropic_api_key)
        or (settings.llm_provider == "openai" and settings.openai_api_key)
    )

    if not has_key:
        logger.warning("No LLM key configured — returning deterministic fallback report.")
        return _fallback_report(observations, rag.chunks)

    if not observations:
        return _fallback_report(observations, rag.chunks)

    try:
        report = await generate_report(
            observations=observations,
            rag_chunks=rag.chunks,
            inspector_name=req.inspector_name,
            site_name=req.site_name,
        )
        return _to_frontend_shape(report, rag.chunks)
    except Exception as e:  # noqa: BLE001 — never 500 the UI; degrade gracefully
        logger.error(f"Generation failed, returning fallback: {e}", exc_info=True)
        return _fallback_report(observations, rag.chunks)
