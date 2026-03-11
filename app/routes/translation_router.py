from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.responses import LegalAIBaseResponse
from app.services.legal_translation_service import LegalTranslationService, LanguageCode


router = APIRouter(prefix="", tags=["translation"])


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate.")
    source_lang: LanguageCode = Field("en", description="Source language code (e.g. 'en', 'hi').")
    target_lang: LanguageCode = Field("hi", description="Target language code (e.g. 'en', 'hi').")


@router.post("/translate-legal-text", response_model=LegalAIBaseResponse)
def translate_legal_text(request: TranslationRequest) -> LegalAIBaseResponse:
    service = LegalTranslationService()
    translated = service.translate(
        text=request.text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
    )
    summary = translated.splitlines()[0] if translated.strip() else ""
    return LegalAIBaseResponse(
        summary=summary,
        legal_sections=[],
        precedents=[],
        analysis="",
        draft=translated,
        safety_flags=[],
    )

