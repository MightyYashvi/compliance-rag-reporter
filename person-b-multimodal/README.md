# Person B — Multimodal Capture & Report Generation

FastAPI backend that handles the "input + output engine" of the Multimodal Field Report Generator.

## What this does

- **Voice → Text**: Whisper transcription (OpenAI API or local model)
- **Image → Text**: Tesseract OCR or LLM vision (Claude/OpenAI) for photos with scene descriptions
- **Text notes**: Direct text input from the field engineer
- **Report generation**: Takes all observations + RAG-retrieved standards → structured, citation-grounded report
- **API contract**: FastAPI endpoints that Person A (RAG) and Person C (Frontend) integrate against

## Quickstart

### 1. Clone & setup

```bash
cd person-b-multimodal
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. System dependencies

```bash
# macOS
brew install tesseract ffmpeg

# Ubuntu/Debian
sudo apt-get install tesseract-ocr ffmpeg libsndfile1

# Windows — download Tesseract installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then add to PATH
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set one of:
#   ANTHROPIC_API_KEY=sk-ant-...   (if using Claude)
#   OPENAI_API_KEY=sk-...          (if using OpenAI / Whisper API)
```

### 4. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API explorer.

### 5. Test

```bash
python test_endpoints.py
```

## Docker

```bash
cp .env.example .env
# edit .env with your keys
docker compose up --build
```

Person B runs on :8000, placeholder RAG on :8001.

## API Endpoints (the contract)

### Capture

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | `/api/capture/text` | JSON `{content, section_hint?}` | `ObservationItem` |
| POST | `/api/capture/voice` | multipart: audio file + section_hint? | `ObservationItem` |
| POST | `/api/capture/image` | multipart: image file + section_hint? | `ObservationItem` |
| POST | `/api/capture/transcribe-raw` | multipart: audio file | `TranscriptionResult` |
| POST | `/api/capture/ocr-raw` | multipart: image file | `OCRResult` |

### Sessions & Reports

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | `/api/sessions` | — | `ObservationSession` |
| GET | `/api/sessions/{id}` | — | `ObservationSession` |
| POST | `/api/sessions/{id}/observations` | `ObservationItem` | `ObservationSession` |
| DELETE | `/api/sessions/{id}/observations/{obs_id}` | — | status |
| POST | `/api/sessions/{id}/generate` | `ReportRequest` | `GeneratedReport` |
| GET | `/api/reports/{id}` | — | `GeneratedReport` |
| GET | `/api/reports` | — | list of report summaries |
| POST | `/api/generate-quick` | list of `ObservationItem` + params | `GeneratedReport` |

### Health

| Method | Endpoint | Output |
|--------|----------|--------|
| GET | `/health` | `{status, version, rag_connected}` |

## Integration with Person A (RAG)

Person B calls Person A's RAG at `POST {RAG_SERVICE_URL}/api/rag/retrieve`.

**Request body** (what we send):
```json
{
  "query": "combined observation text",
  "top_k": 5,
  "filter_tags": null
}
```

**Response body** (what we expect back):
```json
{
  "query": "...",
  "chunks": [
    {
      "chunk_id": "abc-123",
      "text": "The standard says...",
      "source_doc": "OSHA_29CFR1910.pdf",
      "page_number": 42,
      "section_title": "Lockout/Tagout",
      "relevance_score": 0.92,
      "metadata": {}
    }
  ]
}
```

**For development**: set `RAG_MOCK=true` in `.env` to use built-in mock regulatory chunks.

## Integration with Person C (Frontend)

Person C's UI hits these endpoints:

1. **Create session** → `POST /api/sessions`
2. **Capture inputs** → `POST /api/capture/text|voice|image`
3. **Add to session** → `POST /api/sessions/{id}/observations`
4. **Generate** → `POST /api/sessions/{id}/generate`
5. **Get report** → `GET /api/reports/{id}`

The `GeneratedReport` schema includes everything needed for display:
- `sections[]` with `content` and `cited_claims[]`
- Each `CitedClaim` has `source_chunk_ids` and `source_docs` for the provenance layer
- `observations_used` and `rag_chunks_used` for traceability

## Config options

| Env var | Options | Default | Notes |
|---------|---------|---------|-------|
| `LLM_PROVIDER` | `claude`, `openai` | `claude` | Which LLM generates reports |
| `WHISPER_MODE` | `api`, `local` | `api` | `api` needs OPENAI_API_KEY; `local` needs ~1.5GB RAM |
| `OCR_MODE` | `tesseract`, `vision` | `tesseract` | `vision` sends images to the LLM for richer descriptions |
| `RAG_MOCK` | `true`, `false` | `true` | Mock RAG for dev without Person A's service |

## Project structure

```
app/
├── main.py                    # FastAPI entry point
├── config.py                  # Settings from .env
├── models/
│   └── schemas.py             # ★ THE API CONTRACT — Pydantic models
├── services/
│   ├── transcription.py       # Whisper (API + local)
│   ├── ocr.py                 # Tesseract + LLM vision
│   ├── generation.py          # Report generation engine
│   └── rag_client.py          # Person A RAG interface + mock
├── routers/
│   ├── capture.py             # /api/capture/* endpoints
│   └── reports.py             # /api/sessions/* + /api/reports/*
└── prompts/
    └── report_generation.py   # LLM prompt templates
```
