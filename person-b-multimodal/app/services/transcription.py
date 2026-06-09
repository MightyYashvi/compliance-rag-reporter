"""
Whisper transcription service.
Supports two modes:
  - "api"   → OpenAI Whisper API (needs OPENAI_API_KEY)
  - "local" → local faster-whisper model
"""

import os
import tempfile
import logging
from pathlib import Path

from app.config import get_settings
from app.models.schemas import TranscriptionResult

logger = logging.getLogger(__name__)

# Lazy-loaded local model
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel
        settings = get_settings()
        logger.info(f"Loading local Whisper model: {settings.whisper_local_model}")
        _local_model = WhisperModel(
            settings.whisper_local_model,
            device="cpu",       # change to "cuda" if you have GPU
            compute_type="int8"
        )
    return _local_model


async def transcribe_audio(file_path: str, filename: str = "") -> TranscriptionResult:
    """
    Transcribe an audio file using configured Whisper mode.

    Args:
        file_path: Path to the audio file on disk.
        filename: Original filename for logging.

    Returns:
        TranscriptionResult with text, language, duration, confidence.
    """
    settings = get_settings()

    if settings.whisper_mode == "api":
        return await _transcribe_api(file_path, filename)
    else:
        return await _transcribe_local(file_path, filename)


async def _transcribe_api(file_path: str, filename: str) -> TranscriptionResult:
    """Use OpenAI Whisper API."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info(f"Transcribing via OpenAI API: {filename}")

    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            language=None,  # auto-detect
        )

    return TranscriptionResult(
        text=response.text,
        language=getattr(response, "language", None),
        duration_seconds=getattr(response, "duration", None),
        confidence=None,  # Whisper API doesn't return per-segment confidence easily
    )


async def _transcribe_local(file_path: str, filename: str) -> TranscriptionResult:
    """Use local faster-whisper model."""
    logger.info(f"Transcribing locally: {filename}")

    model = _get_local_model()
    segments, info = model.transcribe(file_path, beam_size=5)

    # Collect all segments
    text_parts = []
    total_confidence = 0.0
    segment_count = 0

    for segment in segments:
        text_parts.append(segment.text.strip())
        total_confidence += segment.avg_logprob
        segment_count += 1

    full_text = " ".join(text_parts)
    avg_confidence = (total_confidence / segment_count) if segment_count > 0 else None

    return TranscriptionResult(
        text=full_text,
        language=info.language,
        duration_seconds=info.duration,
        confidence=avg_confidence,
    )
