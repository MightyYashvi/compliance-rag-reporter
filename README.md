# Compliance RAG Reporter — Multimodal Field Report Generator

A system where a field engineer captures site observations and the app generates a regulation-compliant structured draft report.

---

## Team Roles

| Person | Role | Responsibilities |
|--------|------|-----------------|
| Rhea (Person A) | Ingestion & RAG Core | Knowledge base ingestion, chunking, embeddings, ChromaDB, retrieval logic, citation/provenance layer, `POST /rag/retrieve` endpoint |
| Yashvi (Person B) | Multimodal Capture & Generation | Typed notes, Whisper voice transcription, OCR/vision for photos, structured report generation, prompt design |
| Rushika (Person C) | Frontend, Export & Evaluation | React/Streamlit UI, PDF/DOCX export, evaluation harness, Docker deployment |

### Integration Seam

Person B's FastAPI backend is the shared contract.  
Person A exposes retrieval via `POST /rag/retrieve`.  
Person C consumes all backend endpoints in the UI.

---

## Backend Scaffold (Person A — RAG Core)

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

### Running tests

```bash
cd backend
pytest
```

All tests pass without a real OpenAI API key — the vector store, ingestion, and retrieval tests use a fake embedding model injected via dependency parameters.

---

## POST /rag/retrieve — Integration Endpoint for Person B

This is the endpoint Person B's report generation backend calls. Send a field observation as a query; receive ranked regulation chunks with full citation metadata.

**Example curl request**

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

**Example response**

```json
{
  "query": "Corrosion observed near drainage outlet",
  "chunks": [
    {
      "chunk_id": "a3f1b2c4d5e6f7a8",
      "text": "Visible corrosion, rusting, or material degradation near drainage outlets must be documented within 24 hours of discovery.",
      "score": 0.91,
      "source": {
        "document_id": "7e4f1a2b3c4d5e6f",
        "title": "Singapore Drainage Inspection Standard (SDIS)",
        "file_name": "singapore_drainage_inspection_standard.txt",
        "page": 1,
        "section": "2.2",
        "doc_type": "standard",
        "jurisdiction": "Singapore",
        "standard_name": "SDIS"
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

## Retrieval Endpoint Contract

`POST /rag/retrieve`

**Request**
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

**Response**
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
