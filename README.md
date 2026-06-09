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
# fill in OPENAI_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check: `GET http://localhost:8000/health` → `{"status": "ok"}`

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
