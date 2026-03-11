"""
Application configuration for Legal Drafting AI.
Centralizes all configurable settings for easy deployment and integration.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


# Project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
VECTOR_INDEX_DIR = PROJECT_ROOT / "vector_index"
MODELS_DIR = PROJECT_ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "legal_llama_lora"


class Settings(BaseSettings):
    """Application settings loaded from environment or .env."""

    # API
    app_name: str = "Legal Drafting AI"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM (vLLM)
    vllm_base_url: str = Field(
        default="http://localhost:8001/v1",
        description="vLLM server base URL (e.g. http://localhost:8001/v1)",
    )
    llm_model_name: str = "meta-llama/Llama-3-8B-Instruct"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.3
    llm_timeout: float = Field(default=300.0, description="LLM request timeout in seconds (Ollama first load can be slow)")

    # Embeddings (BGE-small or Instructor)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # bge-small; use 768 for instructor-xl
    use_instructor: bool = False  # Set True for instructor-xl

    # RAG
    faiss_index_path: Optional[Path] = None  # Set at runtime or via env
    rag_top_k: int = 3
    rag_score_threshold: float = 0.5
    # Postgres pgvector – AI agent uses its own database (separate from app API and admin-panel).
    use_pgvector: bool = Field(default=False, description="Use legal_ai schema in Postgres for RAG/vectors instead of FAISS")
    database_url: Optional[str] = Field(default=None, description="AI agent DB URL. Prefer LEGAL_AI_AGENT_DATABASE_URL for a dedicated DB, else LEGAL_AI_DATABASE_URL.")

    @model_validator(mode="after")
    def prefer_agent_database_url(self):
        """Use LEGAL_AI_AGENT_DATABASE_URL when set (dedicated AI agent DB)."""
        agent_url = os.environ.get("LEGAL_AI_AGENT_DATABASE_URL")
        if agent_url:
            self.database_url = agent_url
        return self

    # Paths (resolved from PROJECT_ROOT)
    dataset_path: Path = Field(default_factory=lambda: DATASETS_DIR / "legal_drafts.jsonl")
    vector_index_path: Path = Field(default_factory=lambda: VECTOR_INDEX_DIR / "legal_faiss.index")
    vector_metadata_path: Path = Field(default_factory=lambda: VECTOR_INDEX_DIR / "legal_metadata.json")

    # Fine-tuned adapter (optional)
    lora_adapter_path: Optional[Path] = Field(default_factory=lambda: ADAPTER_DIR)

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, description="Default max requests per IP per minute.")
    rate_limit_per_minute_legal_ai: int = Field(default=30, description="Max requests per IP per minute for /legal-ai.")

    class Config:
        env_prefix = "LEGAL_AI_"
        env_file = ".env"
        extra = "ignore"


def get_settings() -> Settings:
    """Return application settings singleton."""
    return Settings()


# Ensure directories exist
VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
