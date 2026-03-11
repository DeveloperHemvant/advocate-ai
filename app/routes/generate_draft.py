"""
API route: POST /generate-draft
Accepts case details and returns generated legal draft.
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.draft_generator import DraftGenerator
from app.services.template_engine import DOCUMENT_TYPES
from app.models.responses import DraftGenerationResponse
from app.services.citation_service import build_citation_lists
from app.services.legal_guardrails import LegalGuardrails
from app.services.legal_retrieval_service import LegalRetrievalService

router = APIRouter(prefix="", tags=["generate"])


class GenerateDraftRequest(BaseModel):
    """Request body for draft generation."""

    document_type: str = Field(
        ...,
        description="One of: bail_application, legal_notice, affidavit, petition, agreement",
    )
    court_name: Optional[str] = Field(None, description="Name of the court")
    client_name: Optional[str] = Field(None, description="Name of client/applicant")
    section: Optional[str] = Field(None, description="Relevant section (e.g. IPC 420)")
    case_facts: str = Field(..., description="Facts of the case / instructions for draft")
    # Optional fields for other document types
    client_address: Optional[str] = None
    opponent_name: Optional[str] = None
    opponent_address: Optional[str] = None
    subject: Optional[str] = None
    father_name: Optional[str] = None
    age: Optional[str] = None
    petition_type: Optional[str] = None

    model_config = {"extra": "allow"}  # Allow extra keys for template placeholders


def _stream_text(text: str):
    chunk_size = 512
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


@router.post("/generate-draft", response_model=DraftGenerationResponse)
def generate_draft(request: GenerateDraftRequest, stream: bool = False):
    """
    Generate a legal draft from case details.
    Uses template + RAG + LLM and returns formatted draft with validation.
    """
    if request.document_type not in DOCUMENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"document_type must be one of: {DOCUMENT_TYPES}",
        )
    payload = request.model_dump()
    case_facts = payload.pop("case_facts")
    document_type = payload.pop("document_type")
    template_placeholders = {k: v for k, v in payload.items() if v is not None}

    generator = DraftGenerator()
    result = generator.generate(
        document_type=document_type,
        case_facts=case_facts,
        extra_context=template_placeholders if template_placeholders else None,
        **template_placeholders,
    )
    draft_text = result["draft"]
    validation = result["validation"]
    success = result["success"]

    # Retrieve sections and judgments for citation enrichment based on the draft content
    retrieval = LegalRetrievalService()
    ctx = retrieval.retrieve_full_context(
        user_query=case_facts,
        document_type=document_type,
        extra_filters={"section": request.section or ""},
    )
    sections = ctx.get("sections") or []
    judgments = ctx.get("judgments") or []
    section_cites, judgment_cites = build_citation_lists(sections, judgments)

    guardrails = LegalGuardrails()
    gr = guardrails.check_citations(section_cites, judgment_cites)

    summary = draft_text.splitlines()[0] if draft_text.strip() else ""

    resp = DraftGenerationResponse(
        summary=summary,
        legal_sections=section_cites,
        precedents=judgment_cites,
        analysis="",
        draft=draft_text,
        validation=validation,
        success=success,
        safety_flags=gr.warnings,
    )
    if stream:
        return StreamingResponse(_stream_text(resp.model_dump_json()), media_type="application/json")
    return resp
