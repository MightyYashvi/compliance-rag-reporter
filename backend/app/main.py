from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.rag_router import router as rag_router
from app.api.generate_router import router as generate_router

load_dotenv()

app = FastAPI(
    title="Compliance RAG Reporter — RAG Core",
    description="Person A: Ingestion, retrieval, and citation layer over a regulatory knowledge base.",
    version="0.1.0",
)

app.include_router(rag_router)
app.include_router(generate_router)


@app.get("/health")
def health():
    return {"status": "ok"}
