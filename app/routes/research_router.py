from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models.legal_llm import LegalLLMClient
from app.models.responses import ResearchResponse
from app.services.citation_service import build_citation_lists
from app.services.intent_classifier import IntentClassifier, LegalIntent
from app.services.legal_guardrails import LegalGuardrails
from app.services.legal_retrieval_service import LegalRetrievalService
from app.utils.prompt_builder import SYSTEM_PROMPT_RESEARCH, build_research_prompt
from app.config import get_settings


router = APIRouter(prefix="", tags=["research"])


class LegalResearchRequest(BaseModel):
    """
    Request body for general legal research.
    """

    query: str = Field(..., description="User's legal research question.")
    jurisdiction: Optional[str] = Field(
        None,
        description="Jurisdiction (e.g. 'Supreme Court', 'Bombay High Court').",
    )
    document_type: Optional[str] = Field(
        None,
        description="Optional document type hint (notice, bail_application, etc.).",
    )


def _stream_text(text: str):
    chunk_size = 512
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


@router.post("/legal-research", response_model=ResearchResponse)
def legal_research(request: LegalResearchRequest, stream: bool = False):
    """
    Legal research endpoint backed by RAG over bare acts, judgments, and drafts.
    """
    classifier = IntentClassifier()
    intent = classifier.classify(request.query)
    if intent != LegalIntent.legal_research:
        # Still proceed, but note in analysis that this looks like another intent.
        pass

    retrieval = LegalRetrievalService()
    ctx = retrieval.retrieve_full_context(
        user_query=request.query,
        document_type=request.document_type,
        extra_filters={"jurisdiction": request.jurisdiction or ""},
    )

    settings = get_settings()
    llm = LegalLLMClient(
        base_url=settings.vllm_base_url,
        model_name=settings.llm_model_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=getattr(settings, "llm_timeout", 300.0),
    )

    graph_context = ctx.get("graph_context") or ""
    combined_context = ctx.get("context_text", "")
    if graph_context:
        combined_context += "\n\n### Graph relationships\n" + graph_context

    prompt = build_research_prompt(
        query=request.query,
        context_text=combined_context,
    )
    answer = llm.complete(
        prompt,
        system_prompt=SYSTEM_PROMPT_RESEARCH,
        temperature=0.2,
    )

    sections = ctx.get("sections") or []
    judgments = ctx.get("judgments") or []
    section_cites, judgment_cites = build_citation_lists(sections, judgments)

    guardrails = LegalGuardrails()
    gr = guardrails.check_citations(section_cites, judgment_cites)

    # Use first lines as summary, rest as analysis
    lines = [l for l in answer.splitlines() if l.strip()]
    summary = lines[0] if lines else ""
    analysis = answer

    resp = ResearchResponse(
        summary=summary,
        legal_sections=section_cites,
        precedents=judgment_cites,
        analysis=analysis,
        draft="",
        retrieved_sections=sections,
        retrieved_judgments=judgments,
        safety_flags=gr.warnings,
    )
    if stream:
        return StreamingResponse(resp.model_dump_json(), media_type="application/json")
    return resp

