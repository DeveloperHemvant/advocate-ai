from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.feedback_service import FeedbackService


router = APIRouter(prefix="", tags=["feedback"])


class FeedbackRequest(BaseModel):
    original_prompt: str = Field(..., description="Original user query or prompt.")
    ai_output: str = Field(..., description="AI-generated output shown to the user.")
    user_corrected_output: Optional[str] = Field(
        None,
        description="User-corrected or edited version of the output.",
    )
    rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Optional rating (1-5).",
    )
    task_type: Optional[str] = Field(
        None,
        description="Type of task (e.g. 'draft_generation', 'legal_research').",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    success: bool
    message: str


@router.post("/ai-feedback", response_model=FeedbackResponse)
def ai_feedback(request: FeedbackRequest) -> FeedbackResponse:
    service = FeedbackService()
    service.save_feedback(
        original_prompt=request.original_prompt,
        ai_output=request.ai_output,
        user_corrected_output=request.user_corrected_output,
        rating=request.rating,
        task_type=request.task_type,
        metadata=request.metadata,
    )
    return FeedbackResponse(success=True, message="Feedback recorded for continuous learning.")

