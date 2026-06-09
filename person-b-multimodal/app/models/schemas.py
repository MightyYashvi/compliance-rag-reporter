"""
============================================================
  API SCHEMAS — THE CONTRACT
  Person A (RAG) and Person C (Frontend) integrate against these.
  Lock this file in week 1. Don't change field names after that.
============================================================
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid


# ── Enums ──────────────────────────────────────────────────

class InputType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"


class ReportStatus(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


# ── Capture / Input Models ─────────────────────────────────

class TextInput(BaseModel):
    """Raw typed field notes from the engineer."""
    content: str = Field(..., min_length=1, description="Field observation text")
    section_hint: Optional[str] = Field(
        None, description="Optional hint about which report section this belongs to"
    )


class TranscriptionResult(BaseModel):
    """Output of Whisper transcription."""
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    confidence: Optional[float] = None


class OCRResult(BaseModel):
    """Output of image OCR / vision analysis."""
    extracted_text: str
    description: Optional[str] = Field(
        None, description="Scene description from vision model (if vision mode)"
    )
    source_filename: str


class ObservationItem(BaseModel):
    """One piece of captured input — text, transcription, or OCR result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_type: InputType
    raw_content: str = Field(..., description="The raw input (text, transcribed text, or OCR text)")
    description: Optional[str] = Field(None, description="Vision description if image")
    source_filename: Optional[str] = None
    section_hint: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── RAG Interface (Person A's contract) ────────────────────

class RAGQuery(BaseModel):
    """What we send to Person A's retrieval endpoint."""
    query: str
    top_k: int = 5
    filter_tags: Optional[list[str]] = None


class RAGChunk(BaseModel):
    """A single retrieved chunk from the knowledge base."""
    chunk_id: str
    text: str
    source_doc: str = Field(..., description="Original document filename or ID")
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[dict] = None


class RAGResponse(BaseModel):
    """What we get back from Person A's retrieval endpoint."""
    query: str
    chunks: list[RAGChunk]


# ── Report Generation ──────────────────────────────────────

class ReportRequest(BaseModel):
    """Request to generate a report from collected observations."""
    observation_session_id: str
    inspector_name: str = "Field Engineer"
    site_name: str = "Site"
    site_location: Optional[str] = None
    report_type: str = Field(
        "field_inspection",
        description="Template type: field_inspection | safety_audit | compliance_check"
    )
    additional_context: Optional[str] = None


class CitedClaim(BaseModel):
    """A single claim in the report with its source provenance."""
    claim_text: str
    source_chunk_ids: list[str] = Field(
        ..., description="IDs of RAG chunks that support this claim"
    )
    source_docs: list[str] = Field(
        ..., description="Original document names for display"
    )
    confidence: Optional[float] = None


class ReportSection(BaseModel):
    """One section of the generated report."""
    title: str
    content: str
    cited_claims: list[CitedClaim] = []


class GeneratedReport(BaseModel):
    """The full generated report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ReportStatus = ReportStatus.COMPLETE
    title: str
    inspector_name: str
    site_name: str
    site_location: Optional[str] = None
    report_date: datetime = Field(default_factory=datetime.utcnow)
    report_type: str
    sections: list[ReportSection]
    observations_used: list[ObservationItem]
    rag_chunks_used: list[RAGChunk] = []
    generation_metadata: Optional[dict] = Field(
        None, description="Model used, tokens, latency, etc."
    )


# ── Observation Session ────────────────────────────────────

class ObservationSession(BaseModel):
    """Groups all observations for one report generation run."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    observations: list[ObservationItem] = []
    report: Optional[GeneratedReport] = None


# ── API Responses ──────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    rag_connected: bool = False


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
