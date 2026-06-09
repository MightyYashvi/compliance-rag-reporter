"""
Prompt templates for structured report generation.
These are the core prompts that turn observations + RAG chunks into reports.
"""

SYSTEM_PROMPT = """You are a professional technical report writer specializing in field inspection reports. You produce regulation-compliant, structured reports based on field observations and relevant standards/regulations.

RULES:
1. Every factual claim about regulations or standards MUST be traceable to a provided source chunk. Do NOT invent or hallucinate regulations.
2. If no relevant standard is provided for an observation, note that no applicable standard was identified in the knowledge base — don't make one up.
3. Use precise, technical language. Avoid vague qualifiers.
4. Structure findings as: observation → applicable standard → assessment → recommendation.
5. Flag any safety-critical findings prominently.
6. When citing a standard, always include the document name and section.

OUTPUT FORMAT:
You must respond in valid JSON matching this structure:
{
  "title": "report title",
  "sections": [
    {
      "title": "section title",
      "content": "section content text",
      "cited_claims": [
        {
          "claim_text": "the specific claim",
          "source_chunk_ids": ["chunk_id_1"],
          "source_docs": ["document_name"],
          "confidence": 0.9
        }
      ]
    }
  ]
}

Do NOT include markdown backticks or any text outside the JSON."""


REPORT_GENERATION_PROMPT = """Generate a structured field inspection report.

## INSPECTOR & SITE
- Inspector: {inspector_name}
- Site: {site_name}
- Location: {site_location}
- Date: {report_date}
- Report Type: {report_type}

## FIELD OBSERVATIONS
The following observations were captured during the field inspection via text notes, voice recordings, and/or photos:

{observations_text}

## RELEVANT STANDARDS & REGULATIONS
The following excerpts from standards/regulations were retrieved from the knowledge base as potentially relevant:

{rag_context}

## INSTRUCTIONS
Generate the report with these sections:
1. **Executive Summary** — 2-3 sentence overview of the inspection and key findings.
2. **Site Information** — Location, date, inspector, conditions.
3. **Observations & Findings** — For each observation: what was seen, applicable standard (cite from the provided chunks), assessment of compliance/condition.
4. **Standards Reference** — List all standards cited, with the specific clauses used.
5. **Recommendations** — Prioritized list of actions, each linked to the finding it addresses.
6. **Conclusion** — Overall compliance status and next steps.

For every claim about a regulation, include a cited_claim entry with the chunk_id(s) from the provided standards. If an observation doesn't match any provided standard, say so explicitly in the finding.

{additional_context}"""


def build_observations_text(observations: list) -> str:
    """Format observation items into a readable block for the prompt."""
    lines = []
    for i, obs in enumerate(observations, 1):
        label = obs.input_type.upper()
        lines.append(f"[Observation {i}] ({label})")
        lines.append(obs.raw_content)
        if obs.description:
            lines.append(f"  Visual description: {obs.description}")
        if obs.section_hint:
            lines.append(f"  Section hint: {obs.section_hint}")
        lines.append("")
    return "\n".join(lines)


def build_rag_context(chunks: list) -> str:
    """Format RAG chunks into a structured context block for the prompt."""
    lines = []
    for chunk in chunks:
        lines.append(f"[Chunk ID: {chunk.chunk_id}]")
        lines.append(f"  Source: {chunk.source_doc}")
        if chunk.page_number:
            lines.append(f"  Page: {chunk.page_number}")
        if chunk.section_title:
            lines.append(f"  Section: {chunk.section_title}")
        lines.append(f"  Relevance: {chunk.relevance_score:.2f}")
        lines.append(f"  Text: {chunk.text}")
        lines.append("")
    return "\n".join(lines)
