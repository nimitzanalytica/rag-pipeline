from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings

# Repo root = same directory as this file
ROOT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: Literal["claude", "ollama"] = "ollama"

    # Anthropic Claude (required when llm_provider=claude)
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-haiku-4-5-20251001"

    # Ollama (required when llm_provider=ollama)
    ollama_model: str = "llama3.2"

    # Shared LLM setting
    max_tokens: int = 1024

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Storage ───────────────────────────────────────────────────────────────
    upload_dir: Path = ROOT_DIR / "uploads"
    chroma_dir: Path = ROOT_DIR / "chroma_db"
    chroma_collection: str = "documents"

    # ── RAG chunking ──────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "rag_pipeline"
    db_user: str = "postgres"
    db_password: str = ""

    class Config:
        env_file = ROOT_DIR / ".env"

    def validate_provider(self):
        """Raise a clear error if Claude is selected but no API key is set."""
        if self.llm_provider == "claude" and not self.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER is set to 'claude' but ANTHROPIC_API_KEY is missing. "
                "Add it to your .env file or switch LLM_PROVIDER to 'ollama'."
            )


settings = Settings()
settings.validate_provider()

settings.upload_dir.mkdir(exist_ok=True)
settings.chroma_dir.mkdir(exist_ok=True)
