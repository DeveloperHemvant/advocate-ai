"""
Lightweight intent classifier for Legal AI requests.
Uses simple keyword rules with optional LLM fallback when ambiguous.
"""

from enum import Enum
from typing import Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient


class LegalIntent(str, Enum):
    draft_generation = "draft_generation"
    legal_research = "legal_research"
    judgment_summary = "judgment_summary"
    case_strategy = "case_strategy"
    document_analysis = "document_analysis"


class IntentClassifier:
    """
    Intent classifier that can be reused across routers.
    """

    def __init__(self, use_llm_fallback: bool = False) -> None:
        self.use_llm_fallback = use_llm_fallback
        self._llm: Optional[LegalLLMClient] = None

    def _get_llm(self) -> LegalLLMClient:
        if self._llm is None:
            settings = get_settings()
            self._llm = LegalLLMClient(
                base_url=settings.vllm_base_url,
                model_name=settings.llm_model_name,
                max_tokens=128,
                temperature=0.0,
                timeout=getattr(settings, "llm_timeout", 60.0),
            )
        return self._llm

    def classify(self, text: str) -> LegalIntent:
        """
        Classify user request into one of the supported intents.
        Uses deterministic keyword rules first; optionally falls back to LLM.
        """
        t = text.lower()

        # Heuristic keywords
        if any(k in t for k in ["draft", "petition", "notice", "bail", "affidavit", "agreement"]):
            return LegalIntent.draft_generation
        if any(k in t for k in ["summarize judgment", "judgment summary", "case law summary", "summarise judgement"]):
            return LegalIntent.judgment_summary
        if any(k in t for k in ["strategy", "case strategy", "how should i proceed", "arguments", "line of argument"]):
            return LegalIntent.case_strategy
        if any(k in t for k in ["analyze document", "analyse document", "review contract", "risky clause", "missing clause"]):
            return LegalIntent.document_analysis

        # Default to research unless LLM classifier is explicitly enabled
        if not self.use_llm_fallback:
            return LegalIntent.legal_research

        # Optional LLM-based refinement
        prompt = (
            "You are an intent classification assistant for an Indian Legal AI system.\n"
            "Classify the user's request into exactly one of the following intents:\n"
            "- draft_generation\n"
            "- legal_research\n"
            "- judgment_summary\n"
            "- case_strategy\n"
            "- document_analysis\n\n"
            "User request:\n"
            f"{text}\n\n"
            "Respond with ONLY the intent name from the list above."
        )
        llm = self._get_llm()
        raw = llm.complete(prompt)
        cleaned = raw.strip().lower()
        for intent in LegalIntent:
            if intent.value == cleaned:
                return intent
        return LegalIntent.legal_research

