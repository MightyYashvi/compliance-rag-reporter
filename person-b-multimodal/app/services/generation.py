"""
Report generation service.
Takes observations + RAG chunks → structured report via LLM.
This is the core of Person B's deliverable.
"""

import json
import time
import logging
from datetime import datetime

from app.config import get_settings
from app.models.schemas import (
    ObservationItem,
    RAGChunk,
    GeneratedReport,
    ReportSection,
    CitedClaim,
    ReportStatus,
)
from app.prompts.report_generation import (
    SYSTEM_PROMPT,
    REPORT_GENERATION_PROMPT,
    build_observations_text,
    build_rag_context,
)

logger = logging.getLogger(__name__)


async def generate_report(
    observations: list[ObservationItem],
    rag_chunks: list[RAGChunk],
    inspector_name: str = "Field Engineer",
    site_name: str = "Site",
    site_location: str | None = None,
    report_type: str = "field_inspection",
    additional_context: str | None = None,
) -> GeneratedReport:
    """
    Generate a structured report from observations and retrieved standards.

    1. Formats observations and RAG context into the prompt
    2. Calls LLM with structured output instructions
    3. Parses the JSON response into a GeneratedReport

    Returns:
        GeneratedReport with sections, cited claims, and metadata.
    """
    settings = get_settings()

    # Build the prompt
    observations_text = build_observations_text(observations)
    rag_context = build_rag_context(rag_chunks)

    user_prompt = REPORT_GENERATION_PROMPT.format(
        inspector_name=inspector_name,
        site_name=site_name,
        site_location=site_location or "Not specified",
        report_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        report_type=report_type,
        observations_text=observations_text,
        rag_context=rag_context if rag_chunks else "No standards retrieved for this inspection.",
        additional_context=additional_context or "",
    )

    # Call the LLM
    start = time.time()

    if settings.llm_provider == "claude":
        raw_response, model_used, tokens_used = await _call_claude(user_prompt)
    else:
        raw_response, model_used, tokens_used = await _call_openai(user_prompt)

    elapsed = time.time() - start
    logger.info(f"Report generated in {elapsed:.1f}s using {model_used}")

    # Parse the LLM output into our schema
    report = _parse_report_json(
        raw_response,
        observations=observations,
        rag_chunks=rag_chunks,
        inspector_name=inspector_name,
        site_name=site_name,
        site_location=site_location,
        report_type=report_type,
        model_used=model_used,
        tokens_used=tokens_used,
        latency=elapsed,
    )

    return report


async def _call_claude(user_prompt: str) -> tuple[str, str, int]:
    """Call Claude API for report generation."""
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return text, settings.claude_model, tokens


async def _call_openai(user_prompt: str) -> tuple[str, str, int]:
    """Call OpenAI API for report generation."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return text, settings.openai_model, tokens


def _parse_report_json(
    raw: str,
    observations: list[ObservationItem],
    rag_chunks: list[RAGChunk],
    inspector_name: str,
    site_name: str,
    site_location: str | None,
    report_type: str,
    model_used: str,
    tokens_used: int,
    latency: float,
) -> GeneratedReport:
    """
    Parse the LLM's JSON output into a GeneratedReport.
    Falls back to a single-section report if JSON parsing fails.
    """
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON: {e}. Using raw text as single section.")
        return GeneratedReport(
            status=ReportStatus.COMPLETE,
            title=f"Field Inspection Report — {site_name}",
            inspector_name=inspector_name,
            site_name=site_name,
            site_location=site_location,
            report_type=report_type,
            sections=[
                ReportSection(
                    title="Report",
                    content=raw,
                    cited_claims=[],
                )
            ],
            observations_used=observations,
            rag_chunks_used=rag_chunks,
            generation_metadata={
                "model": model_used,
                "tokens": tokens_used,
                "latency_seconds": round(latency, 2),
                "parse_error": str(e),
            },
        )

    # Parse sections
    sections = []
    for s in data.get("sections", []):
        cited = []
        for c in s.get("cited_claims", []):
            cited.append(
                CitedClaim(
                    claim_text=c.get("claim_text", ""),
                    source_chunk_ids=c.get("source_chunk_ids", []),
                    source_docs=c.get("source_docs", []),
                    confidence=c.get("confidence"),
                )
            )
        sections.append(
            ReportSection(
                title=s.get("title", "Untitled"),
                content=s.get("content", ""),
                cited_claims=cited,
            )
        )

    return GeneratedReport(
        status=ReportStatus.COMPLETE,
        title=data.get("title", f"Field Inspection Report — {site_name}"),
        inspector_name=inspector_name,
        site_name=site_name,
        site_location=site_location,
        report_type=report_type,
        sections=sections,
        observations_used=observations,
        rag_chunks_used=rag_chunks,
        generation_metadata={
            "model": model_used,
            "tokens": tokens_used,
            "latency_seconds": round(latency, 2),
        },
    )
