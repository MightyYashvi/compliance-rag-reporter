"""Streamlit UI for the Multimodal Field Report Generator.

Capture inputs (typed notes / voice transcript / photo OCR) -> generate a
draft report -> edit it -> review citations -> export to PDF or DOCX.

Run:  streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backend_client import generate_report
from src.export import to_pdf, to_docx

st.set_page_config(page_title="Field Report Generator", layout="wide")

# A small amount of CSS - just spacing and a lighter caption colour.
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 1100px; }
    .ref-box {
        border: 1px solid #e0e0e0; border-radius: 6px;
        padding: 10px 12px; margin-bottom: 8px; background: #fafafa;
    }
    .ref-id { font-weight: 600; color: #1f6feb; }
    .ref-src { color: #666; font-size: 0.85rem; }
    .ref-snip { color: #333; font-size: 0.9rem; font-style: italic; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("Multimodal Field Report Generator")
st.write("Capture site observations from text, voice, or photos and generate "
         "a regulation-grounded draft report with source citations.")
st.divider()

if "report" not in st.session_state:
    st.session_state.report = None

# ---------------- Sidebar: capture inputs ---------------- #
with st.sidebar:
    st.header("1. Capture inputs")
    notes = st.text_area("Typed notes", height=140,
                         placeholder="e.g. Hairline crack on column B3, approx 200mm")
    transcript = st.text_area("Voice transcript", height=100,
                              placeholder="Whisper transcription output")
    ocr_text = st.text_area("Photo OCR / vision text", height=100,
                            placeholder="Text extracted from site photos")
    if st.button("Generate draft", type="primary"):
        with st.spinner("Retrieving standards and drafting report..."):
            st.session_state.report = generate_report({
                "notes": notes, "transcript": transcript, "ocr_text": ocr_text,
            })

report = st.session_state.report

# ---------------- Empty state ---------------- #
if not report:
    st.subheader("How it works")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Capture**\n\nType notes, paste a voice transcript, or add OCR text from photos.")
    c2.markdown("**2. Generate**\n\nEach section is drafted against retrieved regulatory standards.")
    c3.markdown("**3. Verify & export**\n\nCheck the citations, edit inline, and export to PDF or DOCX.")
    st.info("Enter observations in the sidebar and click **Generate draft** to start.")
    st.stop()

# ---------------- Edit + citations ---------------- #
status = report.get("metadata", {}).get("status", "DRAFT")
st.subheader("2. Edit draft")
st.caption(f"Status: {status}")

col_edit, col_cite = st.columns([3, 2], gap="large")

with col_edit:
    report["title"] = st.text_input("Report title", report.get("title", ""))
    for i, sec in enumerate(report["sections"]):
        sec["heading"] = st.text_input(f"Section {i+1} heading", sec["heading"], key=f"h{i}")
        sec["body"] = st.text_area(f"Section {i+1} body", sec["body"], height=130, key=f"b{i}")
        n = len(sec.get("citations", []))
        st.caption(f"{n} source{'s' if n != 1 else ''} linked")

with col_cite:
    st.markdown("**3. Citations / provenance**")
    for ref in report.get("references", []):
        st.markdown(f"""
        <div class="ref-box">
          <span class="ref-id">[{ref['id']}]</span>
          <span class="ref-src">{ref['source']} &middot; chunk {ref['chunk_id']}</span>
          <div class="ref-snip">"{ref.get('snippet','')}"</div>
        </div>""", unsafe_allow_html=True)
    st.caption("Every cited claim links back to a source chunk in the knowledge base.")

# ---------------- Export ---------------- #
st.divider()
st.subheader("4. Export")
e1, e2, _ = st.columns([1, 1, 2])
e1.download_button("Download PDF", data=to_pdf(report),
                   file_name="field_report.pdf", mime="application/pdf")
e2.download_button("Download DOCX", data=to_docx(report),
                   file_name="field_report.docx",
                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
