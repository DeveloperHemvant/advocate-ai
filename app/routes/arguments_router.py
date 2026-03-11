from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.models.responses import ArgumentsResponse
from app.services.citation_service import build_citation_lists
from app.services.legal_guardrails import LegalGuardrails
from app.services.legal_retrieval_service import LegalRetrievalService
from app.utils.prompt_builder import (
    SYSTEM_PROMPT_ARGUMENTS,
    build_arguments_prompt,
)


router = APIRouter(prefix="", tags=["arguments"])


class GenerateArgumentsRequest(BaseModel):
    case_facts: str = Field(..., description="Detailed facts and posture of the case.")
    jurisdiction: Optional[str] = Field(None, description="Court/jurisdiction.")
    section: Optional[str] = Field(None, description="Key provision (e.g. 'NI Act 138').")


@router.post("/generate-arguments", response_model=ArgumentsResponse)
def generate_arguments(request: GenerateArgumentsRequest) -> ArgumentsResponse:
    retrieval = LegalRetrievalService()
    ctx = retrieval.retrieve_full_context(
        user_query=request.case_facts,
        extra_filters={
            "section": request.section or "",
            "jurisdiction": request.jurisdiction or "",
        },
    )
    sections = ctx.get("sections") or []
    judgments = ctx.get("judgments") or []
    section_cites, judgment_cites = build_citation_lists(sections, judgments)

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

    prompt = build_arguments_prompt(
        case_facts=request.case_facts,
        sections_context=sections_context,
        judgments_context=judgments_context,
        jurisdiction=request.jurisdiction,
    )
    answer = llm.complete(
        prompt,
        system_prompt=SYSTEM_PROMPT_ARGUMENTS,
        temperature=0.2,
    )

    # Simple heading-based parsing
    arguments: List[str] = []
    supporting_sections: List[str] = []
    supporting_cases: List[str] = []
    counterarguments: List[str] = []
    current = None
    for line in answer.splitlines():
        lower = line.lower().strip(" :#")
        if lower.startswith("argument"):
            current = "arguments"
            continue
        if "supporting section" in lower or "statutory" in lower:
            current = "sections"
            continue
        if "supporting case" in lower or "case law" in lower:
            current = "cases"
            continue
        if "counterargument" in lower or "counter arguments" in lower:
            current = "counterarguments"
            continue
        if line.strip().startswith(("-", "*")) or line.strip()[:2].isdigit():
            content = line.strip(" -*\t")
            if current == "arguments":
                arguments.append(content)
            elif current == "sections":
                supporting_sections.append(content)
            elif current == "cases":
                supporting_cases.append(content)
            elif current == "counterarguments":
                counterarguments.append(content)

    summary = answer.splitlines()[0] if answer.strip() else ""

    guardrails = LegalGuardrails()
    gr = guardrails.check_citations(section_cites, judgment_cites)

    return ArgumentsResponse(
        summary=summary,
        legal_sections=section_cites,
        precedents=judgment_cites,
        analysis=answer,
        draft="",
        safety_flags=gr.warnings,
        arguments=arguments,
        supporting_sections=supporting_sections,
        supporting_cases=supporting_cases,
        counterarguments=counterarguments,
    )

