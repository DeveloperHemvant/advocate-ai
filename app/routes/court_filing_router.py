from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.responses import LegalAIBaseResponse
from app.services.court_filing_service import CourtFilingService


router = APIRouter(prefix="", tags=["court-filing"])


class CourtDocumentRequest(BaseModel):
    document_type: str = Field(..., description="petition, affidavit, bail_application, legal_notice")
    case_facts: str = Field(..., description="Detailed facts for the document.")
    jurisdiction: str = Field(..., description="Court/forum name.")


@router.post("/generate-court-document", response_model=LegalAIBaseResponse)
def generate_court_document(request: CourtDocumentRequest) -> LegalAIBaseResponse:
    service = CourtFilingService()
    formatted = service.generate_court_document(
        document_type=request.document_type,
        case_facts=request.case_facts,
        jurisdiction=request.jurisdiction,
        extra_context=None,
    )
    summary = formatted.splitlines()[0] if formatted.strip() else ""
    return LegalAIBaseResponse(
        summary=summary,
        legal_sections=[],
        precedents=[],
        analysis="",
        draft=formatted,
        safety_flags=[],
    )

