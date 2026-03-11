"""
Legal Drafting AI - FastAPI application entry point.
Self-hosted legal document generation for Indian advocates.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.routes import (
    admin_router,
    advanced_ai,
    arguments_router,
    case_prediction_router,
    document_router,
    feedback_router,
    generate_draft,
    health,
    judgment_router,
    procedure_router,
    research_router,
    search_legal_docs,
    strategy_router,
    timeline_router,
    translation_router,
    unified_router,
    court_filing_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure paths; shutdown: nothing for now."""
    settings = get_settings()
    settings.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Self-hosted Legal Drafting AI for Indian Advocates. Generate bail applications, legal notices, affidavits, petitions, and agreements using local LLM (Llama 3 8B) and RAG.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Rate limiting
    app.add_middleware(
        RateLimiterMiddleware,
        max_requests=settings.rate_limit_per_minute,
        window_seconds=60,
    )
    app.include_router(health.router)
    app.include_router(advanced_ai.router)
    app.include_router(generate_draft.router)
    app.include_router(search_legal_docs.router)
    app.include_router(research_router.router)
    app.include_router(judgment_router.router)
    app.include_router(strategy_router.router)
    app.include_router(arguments_router.router)
    app.include_router(case_prediction_router.router)
    app.include_router(procedure_router.router)
    app.include_router(court_filing_router.router)
    app.include_router(document_router.router)
    app.include_router(timeline_router.router)
    app.include_router(translation_router.router)
    app.include_router(unified_router.router)
    app.include_router(feedback_router.router)
    app.include_router(admin_router.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
