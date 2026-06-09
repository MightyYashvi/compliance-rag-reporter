"""
Capture router — endpoints for multimodal input ingestion.
  POST /api/capture/text   → text notes
  POST /api/capture/voice  → audio file → Whisper transcription
  POST /api/capture/image  → image file → OCR / vision
"""

import os
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.config import get_settings
from app.models.schemas import (
    TextInput,
    TranscriptionResult,
    OCRResult,
    ObservationItem,
    InputType,
)
from app.services.transcription import transcribe_audio
from app.services.ocr import process_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/capture", tags=["capture"])

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/webm", "audio/ogg", "audio/flac", "audio/mp4", "audio/m4a",
    # browsers sometimes send generic type
    "application/octet-stream",
}

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "image/tiff", "image/bmp",
    "application/octet-stream",
}


def _save_upload(upload: UploadFile, subdir: str) -> str:
    """Save an uploaded file to disk and return the path."""
    settings = get_settings()
    upload_dir = os.path.join(settings.upload_dir, subdir)
    os.makedirs(upload_dir, exist_ok=True)

    ext = Path(upload.filename or "file").suffix or ".bin"
    file_id = uuid.uuid4().hex[:12]
    dest = os.path.join(upload_dir, f"{file_id}{ext}")

    with open(dest, "wb") as f:
        content = upload.file.read()
        f.write(content)

    logger.info(f"Saved upload: {dest} ({len(content)} bytes)")
    return dest


# ── Text Input ─────────────────────────────────────────────

@router.post("/text", response_model=ObservationItem)
async def capture_text(body: TextInput):
    """Capture typed field notes."""
    obs = ObservationItem(
        input_type=InputType.TEXT,
        raw_content=body.content,
        section_hint=body.section_hint,
    )
    return obs


# ── Voice Input ────────────────────────────────────────────

@router.post("/voice", response_model=ObservationItem)
async def capture_voice(
    file: UploadFile = File(..., description="Audio file (mp3, wav, webm, m4a, etc.)"),
    section_hint: str | None = Form(None),
):
    """Upload an audio file → Whisper transcription → ObservationItem."""
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {file.content_type}. "
                   f"Allowed: mp3, wav, webm, ogg, flac, m4a.",
        )

    # Save to disk (Whisper needs a file path)
    file_path = _save_upload(file, "voice")

    try:
        result: TranscriptionResult = await transcribe_audio(file_path, file.filename)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    obs = ObservationItem(
        input_type=InputType.VOICE,
        raw_content=result.text,
        source_filename=file.filename,
        section_hint=section_hint,
    )
    return obs


# ── Image Input ────────────────────────────────────────────

@router.post("/image", response_model=ObservationItem)
async def capture_image(
    file: UploadFile = File(..., description="Image file (jpg, png, etc.)"),
    section_hint: str | None = Form(None),
):
    """Upload an image → OCR/vision → ObservationItem."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. "
                   f"Allowed: jpeg, png, gif, webp, tiff, bmp.",
        )

    file_path = _save_upload(file, "images")

    try:
        result: OCRResult = await process_image(file_path, file.filename)
    except Exception as e:
        logger.error(f"OCR/vision failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

    obs = ObservationItem(
        input_type=InputType.IMAGE,
        raw_content=result.extracted_text,
        description=result.description,
        source_filename=file.filename,
        section_hint=section_hint,
    )
    return obs


# ── Direct transcription/OCR endpoints for testing ─────────

@router.post("/transcribe-raw", response_model=TranscriptionResult)
async def transcribe_raw(file: UploadFile = File(...)):
    """Raw transcription endpoint — returns TranscriptionResult directly."""
    file_path = _save_upload(file, "voice")
    return await transcribe_audio(file_path, file.filename)


@router.post("/ocr-raw", response_model=OCRResult)
async def ocr_raw(file: UploadFile = File(...)):
    """Raw OCR endpoint — returns OCRResult directly."""
    file_path = _save_upload(file, "images")
    return await process_image(file_path, file.filename)
