from typing import Any, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.models.responses import JudgmentSummaryResponse
from app.services.citation_service import build_citation_lists
from app.services.legal_guardrails import LegalGuardrails
from app.services.legal_retrieval_service import LegalRetrievalService
from app.utils.prompt_builder import (
    SYSTEM_PROMPT_JUDGMENT_SUMMARY,
    build_judgment_summary_prompt,
)


router = APIRouter(prefix="", tags=["judgments"])


class SummarizeJudgmentRequest(BaseModel):
    """
    Request body for judgment summarization.
    """

    judgment_text: str = Field(..., description="Full text of the judgment.")
    court: Optional[str] = Field(None, description="Court name, e.g. 'Supreme Court of India'.")
    citation_hint: Optional[str] = Field(
        None,
        description="Known citation or case name (used only for better retrieval).",
    )


class SearchJudgmentsRequest(BaseModel):
    """
    Request for searching judgments.
    """

    query: str = Field(..., description="Search query (facts, issue, party names, etc.).")
    top_k: int = Field(5, ge=1, le=20, description="Number of judgments to retrieve.")


class JudgmentSearchItem(BaseModel):
    case_name: Optional[str] = None
    citation: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchJudgmentsResponse(BaseModel):
    results: List[JudgmentSearchItem]
    count: int


def _split_judgment_summary(text: str) -> dict[str, str]:
    """
    Heuristic splitter that looks for section headings in the LLM output.
    """
    sections = {"facts": "", "issues": "", "decision": "", "ratio": ""}
    current = None
    for line in text.splitlines():
        lower = line.lower().strip(" :#")
        if lower.startswith("facts"):
            current = "facts"
            continue
        if lower.startswith("legal issues") or lower.startswith("issues"):
            current = "issues"
            continue
        if lower.startswith("decision") or lower.startswith("held"):
            current = "decision"
            continue
        if "ratio" in lower:
            current = "ratio"
            continue
        if current:
            sections[current] += line + "\n"
    for k in sections:
        sections[k] = sections[k].strip()
    return sections


@router.post("/summarize-judgment", response_model=JudgmentSummaryResponse)
def summarize_judgment(request: SummarizeJudgmentRequest) -> JudgmentSummaryResponse:
    """
    Summarize a judgment into facts, issues, decision, and ratio.
    """
    retrieval = LegalRetrievalService()
    query = request.citation_hint or request.judgment_text[:500]
    ctx = retrieval.retrieve_full_context(user_query=query)

    settings = get_settings()
    llm = LegalLLMClient(
        base_url=settings.vllm_base_url,
        model_name=settings.llm_model_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=getattr(settings, "llm_timeout", 300.0),
    )

    prompt = build_judgment_summary_prompt(
        judgment_text=request.judgment_text,
        extra_context=ctx.get("context_text", ""),
    )
    answer = llm.complete(
        prompt,
        system_prompt=SYSTEM_PROMPT_JUDGMENT_SUMMARY,
        temperature=0.2,
    )

    sections = ctx.get("sections") or []
    judgments = ctx.get("judgments") or []
    section_cites, judgment_cites = build_citation_lists(sections, judgments)

    guardrails = LegalGuardrails()
    gr = guardrails.check_citations(section_cites, judgment_cites)

    split = _split_judgment_summary(answer)
    summary = split["decision"] or split["facts"] or answer.splitlines()[0]

    return JudgmentSummaryResponse(
        summary=summary,
        legal_sections=section_cites,
        precedents=judgment_cites,
        analysis=answer,
        draft="",
        facts=split["facts"],
        issues=split["issues"],
        decision=split["decision"],
        ratio=split["ratio"],
        safety_flags=gr.warnings,
    )


@router.post("/search-judgments", response_model=SearchJudgmentsResponse)
def search_judgments(request: SearchJudgmentsRequest) -> SearchJudgmentsResponse:
    """
    Semantic search over judgments index.
    """
    retrieval = LegalRetrievalService()
    results = retrieval.get_relevant_judgments(request.query, top_k=request.top_k)
    out: List[JudgmentSearchItem] = []
    for r in results:
        item = JudgmentSearchItem(
            case_name=r.get("case_name"),
            citation=r.get("citation"),
            court=r.get("court"),
            year=r.get("year"),
            score=float(r.get("score", 0.0)),
            metadata={k: v for k, v in r.items() if k not in {"case_name", "citation", "court", "year", "score"}},
        )
        out.append(item)
    return SearchJudgmentsResponse(results=out, count=len(out))

