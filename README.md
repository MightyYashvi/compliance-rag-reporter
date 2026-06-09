# Multimodal Field Report Generator — Capture · Export · Eval

My slice (**Person C**) of the field-report system: the **Streamlit UI**
(capture → edit → review citations), **PDF/DOCX export**, and the
**evaluation harness** that reports faithfulness + citation-coverage metrics.

The RAG core (Person A) and multimodal generation backend (Person B) plug in
behind a single contract. A deterministic mock backend ships here so the UI
and harness run **today** without their services.

## Run the UI

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Capture inputs in the sidebar (typed notes, voice transcript, OCR text),
generate a draft, edit it, review the citation/provenance panel, and export
to PDF or DOCX.

## Run the evaluation harness

```bash
python eval/harness.py
```

Outputs per-case and aggregate metrics:

| Metric | Meaning |
|--------|---------|
| `faithfulness` | lexical grounding of claims in cited source snippets |
| `citation_coverage` | fraction of sections carrying ≥1 citation |
| `keyword_recall` | expected key facts captured in the report |

`score_faithfulness` is offline/CI-friendly; swap it for an LLM-judge later
without touching the rest of the harness.

## Connect the real backend (Person B)

```bash
export FIELD_REPORT_API=http://localhost:8000   # the FastAPI service
```

Contract (lock in week 1): `POST /generate { notes, transcript, ocr_text } -> report`.

## Layout

```
src/app.py             Streamlit capture/edit/cite/export UI
src/export.py          PDF + DOCX renderers          <-- Person C
src/backend_client.py  integration seam (+ mock)
eval/harness.py        faithfulness/coverage metrics <-- Person C headline
eval/test_set.json     test inputs + expected facts
```

## CV line

> Built the capture-to-export UI + an evaluation harness reporting faithfulness
> and citation-coverage metrics over a multimodal RAG report generator.
