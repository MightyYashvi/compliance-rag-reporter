"""
Person B — Multimodal Capture & Report Generation Service
FastAPI backend tying voice/OCR/text inputs → RAG-augmented report generation.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings, ensure_dirs
from app.models.schemas import HealthResponse
from app.routers import capture, reports, generate
from app.services.rag_client import check_rag_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    ensure_dirs()
    logger.info(f"Starting Person B service — LLM: {settings.llm_provider}, Whisper: {settings.whisper_mode}, OCR: {settings.ocr_mode}")

    rag_ok = await check_rag_health()
    if rag_ok:
        logger.info("RAG service connected")
    else:
        logger.warning("RAG service not reachable — using mock data if RAG_MOCK=true")

    yield
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title="Multimodal Field Report Generator — Person B",
    description=(
        "Multimodal capture (text/voice/image) and structured report generation service. "
        "Integrates with Person A's RAG pipeline for regulation-compliant report drafting."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Person C's frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(capture.router)
app.include_router(reports.router)
app.include_router(generate.router)  # frontend contract: POST /generate


@app.get("/health", response_model=HealthResponse)
async def health():
    rag_ok = await check_rag_health()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        rag_connected=rag_ok,
    )


@app.get("/")
async def root():
    return {
        "service": "Person B — Multimodal Capture & Generation",
        "docs": "/docs",
        "health": "/health",
    }
