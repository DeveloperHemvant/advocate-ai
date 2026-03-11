from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.models.responses import CaseStrategyResponse
from app.services.citation_service import build_citation_lists
from app.services.legal_guardrails import LegalGuardrails
from app.services.legal_retrieval_service import LegalRetrievalService
from app.utils.prompt_builder import SYSTEM_PROMPT_STRATEGY, build_case_strategy_prompt


router = APIRouter(prefix="", tags=["strategy"])


class CaseStrategyRequest(BaseModel):
    case_facts: str = Field(..., description="Detailed facts and posture of the case.")
    relevant_section: Optional[str] = Field(
        None,
        description="Known key provision (e.g. 'NI Act 138', 'IPC 420').",
    )
    jurisdiction: Optional[str] = Field(
        None,
        description="Court/jurisdiction where matter will be filed.",
    )


@router.post("/case-strategy", response_model=CaseStrategyResponse)
def case_strategy(request: CaseStrategyRequest) -> CaseStrategyResponse:
    """
    Generate case strategy suggestions based on facts, relevant law, and jurisdiction.
    """
    retrieval = LegalRetrievalService()
    ctx = retrieval.retrieve_full_context(
        user_query=request.case_facts,
        extra_filters={
            "section": request.relevant_section or "",
            "jurisdiction": request.jurisdiction or "",
        },
    )
    sections = ctx.get("sections") or []
    judgments = ctx.get("judgments") or []
    section_cites, judgment_cites = build_citation_lists(sections, judgments)

    guardrails = LegalGuardrails()
    gr = guardrails.check_citations(section_cites, judgment_cites)

    sections_context = "\n".join(section_cites)
    judgments_context = "\n".join(judgment_cites)

    settings = get_settings()
    llm = LegalLLMClient(
        base_url=settings.vllm_base_url,
        model_name=settings.llm_model_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=getattr(settings, "llm_timeout", 300.0),
    )

    prompt = build_case_strategy_prompt(
        case_facts=request.case_facts,
        sections_context=sections_context,
        judgments_context=judgments_context,
        jurisdiction=request.jurisdiction,
    )
    answer = llm.complete(
        prompt,
        system_prompt=SYSTEM_PROMPT_STRATEGY,
        temperature=0.25,
    )

    # Simple split of arguments/procedural steps based on headings or bullets
    arguments: List[str] = []
    procedural: List[str] = []
    current = None
    for line in answer.splitlines():
        lower = line.lower().strip(" :#")
        if "argument" in lower or "submissions" in lower:
            current = "arguments"
            continue
        if "procedure" in lower or "procedural step" in lower or "steps" in lower:
            current = "procedural"
            continue
        if line.strip().startswith(("-", "*")) or line.strip()[:2].isdigit():
            if current == "arguments":
                arguments.append(line.strip(" -*\t"))
            elif current == "procedural":
                procedural.append(line.strip(" -*\t"))

    summary = answer.splitlines()[0] if answer.strip() else ""

    return CaseStrategyResponse(
        summary=summary,
        legal_sections=section_cites,
        precedents=judgment_cites,
        analysis=answer,
        draft="",
        arguments=arguments,
        procedural_steps=procedural,
        safety_flags=gr.warnings,
    )

