from __future__ import annotations

from typing import Literal, Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.utils.prompt_builder import SYSTEM_PROMPT_TRANSLATION, build_translation_prompt


LanguageCode = Literal["en", "hi", "mr", "bn", "ta", "te", "kn", "gu", "pa"]


class LegalTranslationService:
    """
    Uses the local LLM to translate legal text between languages (primarily English/Hindi).
    """

    def __init__(self, llm_client: Optional[LegalLLMClient] = None) -> None:
        settings = get_settings()
        self.llm = llm_client or LegalLLMClient(
            base_url=settings.vllm_base_url,
            model_name=settings.llm_model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=0.2,
            timeout=getattr(settings, "llm_timeout", 120.0),
        )

    def translate(
        self,
        text: str,
        *,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> str:
        prompt = build_translation_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        return self.llm.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT_TRANSLATION,
            temperature=0.2,
        )

