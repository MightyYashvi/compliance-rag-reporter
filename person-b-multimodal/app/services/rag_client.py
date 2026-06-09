"""
RAG Client — interface to Person A's retrieval service.
When RAG_MOCK=true, returns fake regulatory chunks so you can
develop/test Person B's generation without Person A being ready.
"""

import logging
import httpx

from app.config import get_settings
from app.models.schemas import RAGQuery, RAGChunk, RAGResponse

logger = logging.getLogger(__name__)

# ── Mock data for development ──────────────────────────────

MOCK_CHUNKS = [
    RAGChunk(
        chunk_id="mock-001",
        text=(
            "Per OSHA 29 CFR 1910.147, the control of hazardous energy (lockout/tagout) "
            "requires that machines and equipment be isolated from all potentially hazardous "
            "energy sources and locked or tagged out before employees perform servicing or "
            "maintenance activities."
        ),
        source_doc="OSHA_29CFR1910.pdf",
        page_number=42,
        section_title="Lockout/Tagout Requirements",
        relevance_score=0.92,
    ),
    RAGChunk(
        chunk_id="mock-002",
        text=(
            "NFPA 70E Standard for Electrical Safety in the Workplace mandates that "
            "employers perform an arc flash risk assessment before employees work on or "
            "near energized electrical conductors. Appropriate PPE must be selected based "
            "on the incident energy analysis."
        ),
        source_doc="NFPA_70E_2024.pdf",
        page_number=18,
        section_title="Arc Flash Risk Assessment",
        relevance_score=0.87,
    ),
    RAGChunk(
        chunk_id="mock-003",
        text=(
            "ISO 45001:2018 clause 6.1.2 requires organizations to establish processes "
            "for hazard identification that are ongoing and proactive. The identification "
            "of hazards must consider routine and non-routine activities, emergency "
            "situations, and people including contractors and visitors."
        ),
        source_doc="ISO_45001_2018.pdf",
        page_number=9,
        section_title="Hazard Identification",
        relevance_score=0.84,
    ),
    RAGChunk(
        chunk_id="mock-004",
        text=(
            "API 570 Piping Inspection Code requires that inspection intervals for piping "
            "systems be determined based on corrosion rate data. The maximum inspection "
            "interval shall not exceed 10 years for systems with a remaining life exceeding "
            "10 years, or one-half the remaining life for those with less."
        ),
        source_doc="API_570_4th_Edition.pdf",
        page_number=31,
        section_title="Inspection Intervals",
        relevance_score=0.79,
    ),
    RAGChunk(
        chunk_id="mock-005",
        text=(
            "ASTM E2018 Standard Guide for Property Condition Assessments states that "
            "the field observer shall document the physical condition of accessible areas "
            "of the building, including roof, structure, building envelope, mechanical "
            "systems, electrical systems, plumbing, and site conditions."
        ),
        source_doc="ASTM_E2018.pdf",
        page_number=7,
        section_title="Scope of Observation",
        relevance_score=0.75,
    ),
]


async def retrieve_chunks(query: str, top_k: int = 5, filter_tags: list[str] | None = None) -> RAGResponse:
    """
    Retrieve relevant regulatory/standards chunks for the given query.
    Routes to Person A's live service or returns mock data.
    """
    settings = get_settings()

    if settings.rag_mock:
        return _mock_retrieve(query, top_k)
    else:
        return await _live_retrieve(query, top_k, filter_tags)


def _mock_retrieve(query: str, top_k: int) -> RAGResponse:
    """Return mock regulatory chunks for dev/testing."""
    logger.info(f"[MOCK RAG] query='{query}', top_k={top_k}")
    return RAGResponse(
        query=query,
        chunks=MOCK_CHUNKS[:top_k],
    )


async def _live_retrieve(query: str, top_k: int, filter_tags: list[str] | None) -> RAGResponse:
    """
    Call Person A's RAG retrieval endpoint.
    Expected endpoint: POST {RAG_SERVICE_URL}/api/rag/retrieve
    Expected body: RAGQuery
    Expected response: RAGResponse
    """
    settings = get_settings()
    url = f"{settings.rag_service_url}/api/rag/retrieve"

    payload = RAGQuery(query=query, top_k=top_k, filter_tags=filter_tags)

    logger.info(f"[LIVE RAG] POST {url} query='{query}'")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, json=payload.model_dump())
            resp.raise_for_status()
            data = resp.json()
            return RAGResponse(**data)
        except httpx.HTTPError as e:
            logger.error(f"RAG service error: {e}")
            # Fallback to mock if RAG is down
            logger.warning("Falling back to mock RAG data")
            return _mock_retrieve(query, top_k)


async def check_rag_health() -> bool:
    """Ping Person A's RAG service."""
    settings = get_settings()
    if settings.rag_mock:
        return True
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.rag_service_url}/health")
            return resp.status_code == 200
    except Exception:
        return False
