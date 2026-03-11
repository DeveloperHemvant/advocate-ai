from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models.responses import (
    ArgumentsResponse,
    CaseStrategyResponse,
    DocumentAnalysisResponse,
    DraftGenerationResponse,
    JudgmentSummaryResponse,
    LegalAIBaseResponse,
    ProcedureResponse,
    ResearchResponse,
)
from app.routes.arguments_router import generate_arguments, GenerateArgumentsRequest
from app.routes.document_router import analyze_document  # text-based use via DocumentAnalyzer
from app.routes.generate_draft import GenerateDraftRequest, generate_draft
from app.routes.judgment_router import SummarizeJudgmentRequest, summarize_judgment
from app.routes.procedure_router import LegalProcedureRequest, legal_procedure
from app.routes.research_router import LegalResearchRequest, legal_research
from app.routes.strategy_router import CaseStrategyRequest, case_strategy
from app.services.intent_classifier import IntentClassifier, LegalIntent


router = APIRouter(prefix="", tags=["assistant"])


class UnifiedLegalAIRequest(BaseModel):
    query: str = Field(..., description="User's natural language request.")
    intent_hint: Optional[LegalIntent] = Field(
        None,
        description="Optional explicit intent; if omitted, system will infer.",
    )
    jurisdiction: Optional[str] = None
    document_type: Optional[str] = None
    section: Optional[str] = None


def _stream_text(text: str):
    chunk_size = 512
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


@router.post("/legal-ai", response_model=LegalAIBaseResponse)
def legal_ai(request: UnifiedLegalAIRequest, stream: bool = False) -> LegalAIBaseResponse:
    """
    Unified entry point that routes to existing specialized modules.
    """
    classifier = IntentClassifier()
    intent = request.intent_hint or classifier.classify(request.query)

    if intent == LegalIntent.draft_generation:
        draft_req = GenerateDraftRequest(
            document_type=request.document_type or "legal_notice",
            case_facts=request.query,
            section=request.section,
        )
        resp: DraftGenerationResponse = generate_draft(draft_req, stream=False)  # type: ignore[assignment]
        if stream:
            return StreamingResponse(resp.model_dump_json(), media_type="application/json")
        return resp

    if intent == LegalIntent.legal_research:
        research_req = LegalResearchRequest(
            query=request.query,
            jurisdiction=request.jurisdiction,
            document_type=request.document_type,
        )
        resp: ResearchResponse = legal_research(research_req, stream=False)  # type: ignore[assignment]
        if stream:
            return StreamingResponse(resp.model_dump_json(), media_type="application/json")
        return resp

    if intent == LegalIntent.judgment_summary:
        sum_req = SummarizeJudgmentRequest(
            judgment_text=request.query,
            court=request.jurisdiction,
        )
        resp: JudgmentSummaryResponse = summarize_judgment(sum_req)  # type: ignore[assignment]
        if stream:
            return StreamingResponse(resp.model_dump_json(), media_type="application/json")
        return resp

    if intent == LegalIntent.case_strategy:
        strat_req = CaseStrategyRequest(
            case_facts=request.query,
            relevant_section=request.section,
            jurisdiction=request.jurisdiction,
        )
        resp: CaseStrategyResponse = case_strategy(strat_req)  # type: ignore[assignment]
        if stream:
            return StreamingResponse(resp.model_dump_json(), media_type="application/json")
        return resp

    if intent == LegalIntent.document_analysis:
        # For unified endpoint we expect plain text of the document in query
        from app.services.document_analysis_service import DocumentAnalyzer

        analyzer = DocumentAnalyzer()
        answer, ctx, clause_summary = analyzer.analyze(request.query)
        summary = answer.splitlines()[0] if answer.strip() else ""
        resp = DocumentAnalysisResponse(
            summary=summary,
            legal_sections=[],
            precedents=[],
            analysis=answer,
            draft="",
            risky_clauses=clause_summary.get("risky_clauses", []),
            missing_clauses=clause_summary.get("missing_types", []),
            legal_risks=[],
            clause_types=clause_summary.get("present_types", []),
            risk_flags=clause_summary.get("risk_flags", []),
        )
        if stream:
            return StreamingResponse(resp.model_dump_json(), media_type="application/json")
        return resp

    # Fallback: treat as research
    research_req = LegalResearchRequest(
        query=request.query,
        jurisdiction=request.jurisdiction,
        document_type=request.document_type,
    )
    resp = legal_research(research_req, stream=False)
    if stream:
        return StreamingResponse(resp.model_dump_json(), media_type="application/json")
    return resp

