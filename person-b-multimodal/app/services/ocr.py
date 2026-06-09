"""
OCR / Vision service.
Supports two modes:
  - "tesseract" → local Tesseract OCR (fast, text-only)
  - "vision"    → LLM vision API (slower, but gets scene descriptions too)
"""

import base64
import logging
from pathlib import Path

from app.config import get_settings
from app.models.schemas import OCRResult

logger = logging.getLogger(__name__)


async def process_image(file_path: str, filename: str = "") -> OCRResult:
    """
    Extract text (and optionally scene description) from an image.

    Args:
        file_path: Path to the image on disk.
        filename: Original filename.

    Returns:
        OCRResult with extracted_text, optional description, source_filename.
    """
    settings = get_settings()

    if settings.ocr_mode == "vision":
        return await _process_vision(file_path, filename)
    else:
        return await _process_tesseract(file_path, filename)


async def _process_tesseract(file_path: str, filename: str) -> OCRResult:
    """Local Tesseract OCR — text extraction only."""
    import pytesseract
    from PIL import Image

    logger.info(f"OCR via Tesseract: {filename}")

    img = Image.open(file_path)

    # Preprocess: convert to grayscale for better OCR
    if img.mode != "L":
        img = img.convert("L")

    text = pytesseract.image_to_string(img, lang="eng")

    return OCRResult(
        extracted_text=text.strip(),
        description=None,
        source_filename=filename,
    )


async def _process_vision(file_path: str, filename: str) -> OCRResult:
    """
    LLM vision — sends the image to Claude or OpenAI vision API.
    Gets both text extraction AND a scene description.
    """
    settings = get_settings()

    # Read image as base64
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Detect media type
    suffix = Path(file_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    if settings.llm_provider == "claude":
        return await _vision_claude(image_data, media_type, filename)
    else:
        return await _vision_openai(image_data, media_type, filename)


async def _vision_claude(image_data: str, media_type: str, filename: str) -> OCRResult:
    """Use Claude's vision to analyze the image."""
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    logger.info(f"Vision via Claude: {filename}")

    prompt = (
        "You are analyzing a field inspection photo. Do two things:\n"
        "1. Extract ALL visible text (signs, labels, readings, markings). "
        "Output under 'EXTRACTED TEXT:'.\n"
        "2. Describe what you see that's relevant to an inspection report "
        "(equipment condition, hazards, environment). Output under 'SCENE DESCRIPTION:'.\n"
        "Be precise and factual. No speculation."
    )

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    result_text = response.content[0].text

    # Parse the structured output
    extracted = ""
    description = ""

    if "EXTRACTED TEXT:" in result_text:
        parts = result_text.split("SCENE DESCRIPTION:")
        extracted = parts[0].split("EXTRACTED TEXT:")[-1].strip()
        if len(parts) > 1:
            description = parts[1].strip()
    else:
        extracted = result_text
        description = result_text

    return OCRResult(
        extracted_text=extracted,
        description=description,
        source_filename=filename,
    )


async def _vision_openai(image_data: str, media_type: str, filename: str) -> OCRResult:
    """Use OpenAI GPT-4V to analyze the image."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info(f"Vision via OpenAI: {filename}")

    prompt = (
        "You are analyzing a field inspection photo. Do two things:\n"
        "1. Extract ALL visible text (signs, labels, readings, markings). "
        "Output under 'EXTRACTED TEXT:'.\n"
        "2. Describe what you see that's relevant to an inspection report "
        "(equipment condition, hazards, environment). Output under 'SCENE DESCRIPTION:'.\n"
        "Be precise and factual. No speculation."
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    result_text = response.choices[0].message.content

    extracted = ""
    description = ""

    if "EXTRACTED TEXT:" in result_text:
        parts = result_text.split("SCENE DESCRIPTION:")
        extracted = parts[0].split("EXTRACTED TEXT:")[-1].strip()
        if len(parts) > 1:
            description = parts[1].strip()
    else:
        extracted = result_text
        description = result_text

    return OCRResult(
        extracted_text=extracted,
        description=description,
        source_filename=filename,
    )
