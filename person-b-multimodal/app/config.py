from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "claude"
    claude_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"

    # Whisper
    whisper_mode: str = "api"  # "local" or "api"
    whisper_local_model: str = "base"

    # OCR
    ocr_mode: str = "tesseract"  # "tesseract" or "vision"

    # RAG
    rag_service_url: str = "http://localhost:8001"
    rag_mock: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Storage
    upload_dir: str = "./uploads"
    reports_dir: str = "./reports"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def ensure_dirs():
    """Create upload/report dirs on startup."""
    s = get_settings()
    os.makedirs(s.upload_dir, exist_ok=True)
    os.makedirs(s.reports_dir, exist_ok=True)
