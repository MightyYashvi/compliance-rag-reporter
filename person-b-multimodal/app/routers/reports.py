"""
Reports router — observation sessions and report generation.
  POST /api/sessions                     → create session
  POST /api/sessions/{id}/observations   → add observation to session
  POST /api/sessions/{id}/generate       → generate report from session
  GET  /api/reports/{id}                 → get generated report
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ObservationItem,
    ObservationSession,
    ReportRequest,
    GeneratedReport,
    ReportStatus,
)
from app.services.rag_client import retrieve_chunks
from app.services.generation import generate_report

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reports"])

# ── In-memory store (swap for Redis/DB in prod) ────────────
sessions: dict[str, ObservationSession] = {}
reports: dict[str, GeneratedReport] = {}


# ── Sessions ───────────────────────────────────────────────

@router.post("/api/sessions", response_model=ObservationSession)
async def create_session():
    """Create a new observation session."""
    session = ObservationSession()
    sessions[session.session_id] = session
    logger.info(f"Created session: {session.session_id}")
    return session


@router.get("/api/sessions/{session_id}", response_model=ObservationSession)
async def get_session(session_id: str):
    """Get session with all its observations."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@router.post("/api/sessions/{session_id}/observations", response_model=ObservationSession)
async def add_observation(session_id: str, observation: ObservationItem):
    """
    Add a processed observation to a session.
    Typically called after /api/capture/* returns an ObservationItem.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    session.observations.append(observation)
    logger.info(
        f"Session {session_id}: added {observation.input_type} observation "
        f"(total: {len(session.observations)})"
    )
    return session


@router.delete("/api/sessions/{session_id}/observations/{obs_id}")
async def remove_observation(session_id: str, obs_id: str):
    """Remove an observation from a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    before = len(session.observations)
    session.observations = [o for o in session.observations if o.id != obs_id]

    if len(session.observations) == before:
        raise HTTPException(status_code=404, detail="Observation not found in session")

    return {"removed": obs_id, "remaining": len(session.observations)}


# ── Report Generation ──────────────────────────────────────

@router.post("/api/sessions/{session_id}/generate", response_model=GeneratedReport)
async def generate_from_session(session_id: str, request: ReportRequest):
    """
    Generate a structured report from all observations in this session.

    Flow:
    1. Collects all observations from the session
    2. Builds a combined query from observations
    3. Calls Person A's RAG to retrieve relevant standards
    4. Passes observations + RAG chunks to the generation engine
    5. Returns the structured report with citations
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if not session.observations:
        raise HTTPException(status_code=400, detail="No observations in session. Add observations first.")

    # Build a combined query from all observations for RAG retrieval
    combined_query = _build_retrieval_query(session.observations)
    logger.info(f"RAG query: {combined_query[:200]}...")

    # Retrieve relevant standards/regulations via Person A's RAG
    rag_response = await retrieve_chunks(query=combined_query, top_k=5)

    # Generate the report
    try:
        report = await generate_report(
            observations=session.observations,
            rag_chunks=rag_response.chunks,
            inspector_name=request.inspector_name,
            site_name=request.site_name,
            site_location=request.site_location,
            report_type=request.report_type,
            additional_context=request.additional_context,
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    # Store the report
    reports[report.report_id] = report
    session.report = report

    return report


@router.get("/api/reports/{report_id}", response_model=GeneratedReport)
async def get_report(report_id: str):
    """Retrieve a previously generated report."""
    if report_id not in reports:
        raise HTTPException(status_code=404, detail="Report not found")
    return reports[report_id]


@router.get("/api/reports", response_model=list[dict])
async def list_reports():
    """List all generated reports (summary only)."""
    return [
        {
            "report_id": r.report_id,
            "title": r.title,
            "site_name": r.site_name,
            "inspector_name": r.inspector_name,
            "report_date": r.report_date.isoformat(),
            "status": r.status,
            "sections_count": len(r.sections),
        }
        for r in reports.values()
    ]


# ── Quick generate (no session needed) ─────────────────────

@router.post("/api/generate-quick", response_model=GeneratedReport)
async def generate_quick(
    observations: list[ObservationItem],
    inspector_name: str = "Field Engineer",
    site_name: str = "Site",
    site_location: str | None = None,
    report_type: str = "field_inspection",
):
    """
    One-shot report generation — pass observations directly without a session.
    Useful for testing and simple cases.
    """
    if not observations:
        raise HTTPException(status_code=400, detail="At least one observation is required")

    combined_query = _build_retrieval_query(observations)
    rag_response = await retrieve_chunks(query=combined_query, top_k=5)

    report = await generate_report(
        observations=observations,
        rag_chunks=rag_response.chunks,
        inspector_name=inspector_name,
        site_name=site_name,
        site_location=site_location,
        report_type=report_type,
    )

    reports[report.report_id] = report
    return report


# ── Helpers ────────────────────────────────────────────────

def _build_retrieval_query(observations: list[ObservationItem]) -> str:
    """
    Combine observation texts into a single retrieval query.
    Keeps it under ~500 tokens to avoid RAG query length issues.
    """
    parts = []
    for obs in observations:
        text = obs.raw_content.strip()
        if text:
            parts.append(text[:300])  # cap each observation
    combined = " | ".join(parts)
    return combined[:2000]  # hard cap
