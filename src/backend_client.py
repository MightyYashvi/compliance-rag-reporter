"""Backend client  —  the integration seam with Person B's FastAPI.

The UI calls these functions. Set FIELD_REPORT_API to the real backend URL
when Person B's service is up; otherwise it falls back to a deterministic
mock so the UI + eval harness run today.

Contract (lock in week 1):
  POST /generate  { notes, transcript, ocr_text }  ->  report dict
"""
from __future__ import annotations

import os
from typing import Any

import requests

# Defaults to the deployed Hugging Face Space backend; override with the
# FIELD_REPORT_API env var / Streamlit secret to point elsewhere (or set it
# empty to force the local mock).
_DEFAULT_API = "https://mightyashvi-compliance-rag-backend.hf.space"
API_URL = os.environ.get("FIELD_REPORT_API", _DEFAULT_API).rstrip("/")


def _mock_generate(inputs: dict[str, str]) -> dict[str, Any]:
    """Deterministic mock report so the UI is runnable without the backend."""
    notes = inputs.get("notes", "")
    transcript = inputs.get("transcript", "")
    ocr = inputs.get("ocr_text", "")
    combined = " ".join(x for x in [notes, transcript, ocr] if x).strip()

    return {
        "title": "Site Inspection Field Report",
        "metadata": {"inspector": "R. (auto)", "status": "DRAFT"},
        "sections": [
            {
                "heading": "Observations",
                "body": combined or "No observations captured.",
                "citations": [{"id": 1, "source": "Reg-Standard-A.pdf",
                               "chunk_id": "c12"}],
            },
            {
                "heading": "Compliance Assessment",
                "body": ("Structural elements were inspected per the quarterly "
                         "requirement. Non-conforming items must be logged "
                         "within 24 hours; the flagged item was logged for "
                         "review per clause 4.2."),
                "citations": [
                    {"id": 1, "source": "Reg-Standard-A.pdf", "chunk_id": "c12"},
                    {"id": 2, "source": "Reg-Standard-A.pdf", "chunk_id": "c08"},
                ],
            },
        ],
        "references": [
            {"id": 1, "source": "Reg-Standard-A.pdf", "chunk_id": "c12",
             "snippet": "All structural elements shall be inspected quarterly."},
            {"id": 2, "source": "Reg-Standard-A.pdf", "chunk_id": "c08",
             "snippet": "Non-conforming items must be logged within 24 hours."},
        ],
    }


def generate_report(inputs: dict[str, str]) -> dict[str, Any]:
    """Call the real backend if configured, else the mock."""
    if not API_URL:
        return _mock_generate(inputs)
    resp = requests.post(f"{API_URL}/generate", json=inputs, timeout=60)
    resp.raise_for_status()
    return resp.json()
