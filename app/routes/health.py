"""
Health and readiness checks for the Legal Drafting AI API.
"""

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(prefix="", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class ReadinessResponse(BaseModel):
    ready: bool
    vllm_configured: bool
    vector_index_configured: bool


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", app="Legal Drafting AI", version="1.0.0")


@router.get("/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    """Readiness: checks that config and paths are set (does not call vLLM or FAISS)."""
    settings = get_settings()
    return ReadinessResponse(
        ready=True,
        vllm_configured=bool(settings.vllm_base_url),
        vector_index_configured=Path(settings.vector_index_path).exists() if settings.vector_index_path else False,
    )
