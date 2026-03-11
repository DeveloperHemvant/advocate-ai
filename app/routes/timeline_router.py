from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.responses import LegalAIBaseResponse
from app.services.timeline_service import TimelineEvent, TimelineService


router = APIRouter(prefix="", tags=["timeline"])


class TimelineRequest(BaseModel):
    case_text: str = Field(..., description="Full case narrative or key facts.")


class TimelineEventModel(BaseModel):
    label: str
    date: str | None
    description: str


class TimelineResponse(LegalAIBaseResponse):
    events: List[TimelineEventModel] = Field(default_factory=list)


@router.post("/extract-case-timeline", response_model=TimelineResponse)
def extract_case_timeline(request: TimelineRequest) -> TimelineResponse:
    service = TimelineService()
    events = service.extract_events(request.case_text)
    models = [TimelineEventModel(label=e.label, date=e.date, description=e.description) for e in events]
    summary = " -> ".join(f"{e.label}({e.date or '?'})" for e in events) if events else ""
    return TimelineResponse(
        summary=summary,
        legal_sections=[],
        precedents=[],
        analysis="",
        draft="",
        safety_flags=[],
        events=models,
    )

