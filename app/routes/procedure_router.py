from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.responses import ProcedureResponse
from app.services.citation_service import build_citation_lists
from app.services.legal_guardrails import LegalGuardrails
from app.services.legal_retrieval_service import LegalRetrievalService
from app.services.procedure_service import ProcedureEngine


router = APIRouter(prefix="", tags=["procedure"])


class LegalProcedureRequest(BaseModel):
    legal_issue: str = Field(..., description="Description of the legal issue (e.g. 'cheque bounce under NI Act 138').")
    jurisdiction: Optional[str] = Field(
        None,
        description="Court or forum (e.g. 'Metropolitan Magistrate', 'District Consumer Commission').",
    )


@router.post("/legal-procedure", response_model=ProcedureResponse)
def legal_procedure(request: LegalProcedureRequest) -> ProcedureResponse:
    engine = ProcedureEngine()
    raw, ctx = engine.explain_procedure(request.legal_issue, request.jurisdiction)

    # Parse steps, documents, timeline heuristically
    steps: List[str] = []
    documents: List[str] = []
    timeline = ""
    current = None
    for line in raw.splitlines():
        lower = line.lower().strip(" :#")
        if "step" in lower or "procedure" in lower:
            current = "steps"
            continue
        if "document" in lower:
            current = "documents"
            continue
        if "timeline" in lower or "time" in lower or "limitation" in lower:
            current = "timeline"
            continue
        if line.strip().startswith(("-", "*")) or line.strip()[:2].isdigit():
            content = line.strip(" -*\t")
            if current == "steps":
                steps.append(content)
            elif current == "documents":
                documents.append(content)
        elif current == "timeline":
            timeline += line + "\n"
    timeline = timeline.strip()

    sections = ctx.get("sections") or []
    judgments = ctx.get("judgments") or []
    section_cites, judgment_cites = build_citation_lists(sections, judgments)

    summary = raw.splitlines()[0] if raw.strip() else ""

    guardrails = LegalGuardrails()
    gr = guardrails.check_citations(section_cites, judgment_cites)

    return ProcedureResponse(
        summary=summary,
        legal_sections=section_cites,
        precedents=judgment_cites,
        analysis=raw,
        draft="",
        safety_flags=gr.warnings,
        steps=steps,
        documents_required=documents,
        timeline=timeline,
    )

