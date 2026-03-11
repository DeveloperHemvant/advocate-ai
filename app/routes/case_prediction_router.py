from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.responses import LegalAIBaseResponse
from app.services.case_prediction_service import CasePredictionService
from app.services.legal_guardrails import LegalGuardrails


router = APIRouter(prefix="", tags=["prediction"])


class CaseOutcomeRequest(BaseModel):
    case_facts: str = Field(..., description="Detailed facts and posture of the case.")
    section: Optional[str] = Field(None, description="Key statutory provision involved.")
    court_level: str = Field(..., description="Court level (e.g. Magistrate, Sessions Court, High Court, Supreme Court).")


class CaseOutcomeResponse(LegalAIBaseResponse):
    success_probability: float = Field(..., ge=0.0, le=1.0)
    similar_cases: List[dict] = Field(default_factory=list)
    key_factors: List[str] = Field(default_factory=list)
    risk_assessment: str = ""


@router.post("/predict-case-outcome", response_model=CaseOutcomeResponse)
def predict_case_outcome(request: CaseOutcomeRequest) -> CaseOutcomeResponse:
    service = CasePredictionService()
    raw, meta = service.predict_outcome(
        case_facts=request.case_facts,
        section=request.section,
        court_level=request.court_level,
    )

    # Heuristic extraction from LLM answer
    success_prob = 0.5
    key_factors: List[str] = []
    risk_assessment = ""
    current = None
    for line in raw.splitlines():
        lower = line.lower()
        if "probability" in lower or "%" in lower:
            # Extract first percentage number
            import re

            m = re.search(r"(\d{1,3})\s*%", line)
            if m:
                val = int(m.group(1))
                success_prob = max(0.0, min(1.0, val / 100.0))
        if "key factor" in lower or "favourable factor" in lower or "unfavourable factor" in lower:
            current = "factors"
            continue
        if "risk assessment" in lower:
            current = "risk"
            continue
        if line.strip().startswith(("-", "*")) and current == "factors":
            key_factors.append(line.strip(" -*\t"))
        elif current == "risk":
            risk_assessment += line + "\n"
    risk_assessment = risk_assessment.strip()

    sections = meta.get("sections") or []
    judgments = meta.get("judgments") or []
    section_cites = meta.get("section_citations") or []
    judgment_cites = meta.get("judgment_citations") or []

    guardrails = LegalGuardrails()
    gr = guardrails.check_citations(section_cites, judgment_cites)

    summary = raw.splitlines()[0] if raw.strip() else ""

    return CaseOutcomeResponse(
        summary=summary,
        legal_sections=section_cites,
        precedents=judgment_cites,
        analysis=raw,
        draft="",
        safety_flags=gr.warnings,
        success_probability=success_prob,
        similar_cases=judgments,
        key_factors=key_factors,
        risk_assessment=risk_assessment,
    )

