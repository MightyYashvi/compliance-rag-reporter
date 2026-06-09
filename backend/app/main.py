from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Compliance RAG Reporter — RAG Core",
    description="Person A: Ingestion, retrieval, and citation layer over a regulatory knowledge base.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}
