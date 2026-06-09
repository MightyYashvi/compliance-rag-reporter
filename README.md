# Compliance RAG Reporter — Multimodal Field Report Generator

A system where a field engineer captures site observations and the app generates a regulation-compliant structured draft report.

---

## Team Roles

| Person | Role | Responsibilities |
|--------|------|-----------------|
| Rhea (Person A) | Ingestion & RAG Core | Knowledge base ingestion, chunking, embeddings, ChromaDB, retrieval logic, citation/provenance layer, `POST /rag/retrieve` endpoint |
| Yashvi (Person B) | Multimodal Capture & Generation | Typed notes, Whisper voice transcription, OCR/vision for photos, structured report generation, prompt design |
| Rushika (Person C) | Frontend, Export & Evaluation | Streamlit UI, PDF/DOCX export, evaluation harness, Docker deployment |

### Integration Seam

Person B's FastAPI backend is the shared contract.
Person A exposes retrieval via `POST /rag/retrieve`.
Person C consumes all backend endpoints in the UI and runs the evaluation harness.

---

## Backend — RAG Core (Person A)

```
backend/
├── app/
│   ├── main.py          # FastAPI app entry point
│   ├── api/             # Route modules (rag retrieval endpoint goes here)
│   └── rag/             # Core RAG logic: loaders, chunkers, embeddings, retriever
├── data/
│   ├── kb_raw/          # Raw regulatory documents (PDF, DOCX) — not committed
│   └── chroma_db/       # ChromaDB persistent vector store — not committed
├── scripts/             # One-off ingestion and utility scripts
├── tests/               # pytest test suite
├── requirements.txt
├── .env.example
└── .gitignore
```

### Running locally

```bash
cd backend
cp .env.example .env
# fill in OPENAI_API_KEY and CHROMA_PERSIST_DIR
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check: `GET http://localhost:8000/health` → `{"status": "ok"}`

### Ingesting the knowledge base

Run once before starting the server to embed all KB documents into ChromaDB:

```bash
cd backend
python scripts/ingest.py

# Optional flags:
# --reset            clear existing collection first
# --source-dir PATH  override kb_raw location
# --chunk-size N     characters per chunk (default: 1000)
# --chunk-overlap N  overlap between chunks (default: 200)
```

### How embeddings and semantic search work

**Embeddings** convert regulation text into dense numerical vectors. The model (`text-embedding-3-small`) is trained so that texts with similar meaning produce nearby vectors — even when they share no exact words. A field engineer writing *"rust forming near a pipe joint"* retrieves chunks about *"corrosion near drainage outlets"* because both map to similar regions in the embedding space.

**ChromaDB** stores each chunk as a `(vector, text, metadata)` triple on disk. At query time it computes cosine similarity between the query vector and all stored chunk vectors, returning the top-K closest matches. This is fundamentally different from keyword search, which requires exact term overlap.

**Why this matters for citations:** every chunk stored in ChromaDB carries the full provenance metadata attached by the loader and chunker — document title, standard name, section number, jurisdiction. The retrieval endpoint returns this metadata alongside the chunk text so the report generator can produce source-grounded citations without any secondary lookup.

### End-to-end smoke test (requires OPENAI_API_KEY)

```bash
cd backend
python scripts/smoke_test.py           # ingest (if not done) then query
python scripts/smoke_test.py --reset   # wipe collection and re-ingest first
```

### Running unit tests (no API key needed)

```bash
cd backend
pytest
```

All 93 tests pass without a real OpenAI API key — vector store, ingestion, and retrieval tests use a fake embedding model injected via dependency parameters.

---

## POST /rag/retrieve — Integration Endpoint for Person B

Send a field observation as a query; receive ranked regulation chunks with full citation metadata.

```bash
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Corrosion observed near drainage outlet",
    "top_k": 5,
    "filters": {
      "doc_type": "standard",
      "jurisdiction": "Singapore"
    }
  }'
```

**Request schema**
```json
{
  "query": "Corrosion observed near drainage outlet",
  "top_k": 5,
  "filters": {
    "doc_type": "standard",
    "jurisdiction": "Singapore",
    "standard_name": "optional"
  }
}
```

**Response schema**
```json
{
  "query": "Corrosion observed near drainage outlet",
  "chunks": [
    {
      "chunk_id": "string",
      "text": "retrieved regulation chunk",
      "score": 0.87,
      "source": {
        "document_id": "string",
        "title": "string",
        "file_name": "string",
        "page": 1,
        "section": "2.2",
        "doc_type": "standard",
        "jurisdiction": "Singapore",
        "standard_name": "string"
      }
    }
  ]
}
```

**Error responses**

| Code | Condition |
|------|-----------|
| `422` | Query is blank, `top_k` is 0 or > 20 |
| `503` | Vector store empty (run ingestion first) or `OPENAI_API_KEY` not set |

---

## Frontend, Export & Evaluation — Person C

```
src/app.py             Streamlit capture/edit/cite/export UI
src/export.py          PDF + DOCX renderers
src/backend_client.py  integration seam (+ mock)
eval/harness.py        faithfulness/coverage metrics
eval/test_set.json     test inputs + expected facts
```

### Run the UI

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Capture inputs in the sidebar (typed notes, voice transcript, OCR text), generate a draft, edit it, review the citation/provenance panel, and export to PDF or DOCX.

### Run the evaluation harness

```bash
python eval/harness.py
```

Outputs per-case and aggregate metrics:

| Metric | Meaning |
|--------|---------|
| `faithfulness` | lexical grounding of claims in cited source snippets |
| `citation_coverage` | fraction of sections carrying ≥1 citation |
| `keyword_recall` | expected key facts captured in the report |

`score_faithfulness` is offline/CI-friendly; swap it for an LLM-judge later without touching the rest of the harness.

### Connect the real backend (Person B)

```bash
export FIELD_REPORT_API=http://localhost:8000   # the FastAPI service
```

Contract: `POST /generate { notes, transcript, ocr_text } -> report`

A deterministic mock backend ships with Person C's code so the UI and harness run without the live services.
