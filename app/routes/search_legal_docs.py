"""
API route: search legal documents (RAG retrieval).
Useful for finding similar past drafts without generating.
"""

from typing import Any, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.services.rag_service import RAGService

router = APIRouter(prefix="", tags=["search"])


class SearchLegalDocsRequest(BaseModel):
    """Request body for semantic search over legal drafts."""

    document_type: str = Field(..., description="Filter by document type")
    query: str = Field(..., description="Search query / case facts")
    top_k: Optional[int] = Field(3, ge=1, le=20, description="Number of results")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Min similarity score")


class SearchResultItem(BaseModel):
    """Single search result with draft excerpt and score."""

    document_type: str
    facts: Optional[str] = None
    draft_excerpt: str
    score: float


class SearchLegalDocsResponse(BaseModel):
    """Response with list of similar drafts."""

    results: list[dict[str, Any]]
    count: int


@router.post("/search-legal-docs", response_model=SearchLegalDocsResponse)
def search_legal_docs(request: SearchLegalDocsRequest) -> SearchLegalDocsResponse:
    """
    Search for similar legal drafts by document type and query.
    Returns retrieved examples from the FAISS index (for RAG or reference).
    """
    rag = RAGService()
    results = rag.retrieve(
        document_type=request.document_type,
        case_facts=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )
    # Normalize for response: include excerpt of draft
    out = []
    for r in results:
        draft = r.get("draft", "")
        out.append({
            "document_type": r.get("document_type", request.document_type),
            "facts": r.get("facts"),
            "draft_excerpt": draft[:1500] + ("..." if len(draft) > 1500 else ""),
            "score": r.get("score", 0.0),
        })
    return SearchLegalDocsResponse(results=out, count=len(out))


@router.get("/search-legal-docs")
def search_legal_docs_get(
    document_type: str = Query(..., description="Document type"),
    query: str = Query(..., description="Search query"),
    top_k: int = Query(3, ge=1, le=20),
) -> SearchLegalDocsResponse:
    """GET variant of search for convenience."""
    return search_legal_docs(SearchLegalDocsRequest(document_type=document_type, query=query, top_k=top_k))
