from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.services.citation_service import build_citation_lists
from app.services.legal_retrieval_service import LegalRetrievalService
from app.utils.prompt_builder import SYSTEM_PROMPT_STRATEGY, build_case_strategy_prompt


class CasePredictionService:
    """
    Estimates case outcome probabilities using similar judgments and sections.
    """

    def __init__(
        self,
        retrieval_service: Optional[LegalRetrievalService] = None,
        llm_client: Optional[LegalLLMClient] = None,
    ) -> None:
        self.retrieval = retrieval_service or LegalRetrievalService()
        settings = get_settings()
        self.llm = llm_client or LegalLLMClient(
            base_url=settings.vllm_base_url,
            model_name=settings.llm_model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout=getattr(settings, "llm_timeout", 300.0),
        )

    def _build_prediction_prompt(
        self,
        case_facts: str,
        court_level: str,
        sections_context: str,
        judgments_context: str,
    ) -> str:
        base_prompt = build_case_strategy_prompt(
            case_facts=case_facts,
            sections_context=sections_context,
            judgments_context=judgments_context,
            jurisdiction=court_level,
        )
        extra = (
            "\n\nAdditionally, estimate the probability of success for the client as a percentage "
            "at the specified court level, identify key favourable and unfavourable factors, and "
            "provide a concise risk assessment."
        )
        return base_prompt + extra

    def predict_outcome(
        self,
        case_facts: str,
        section: Optional[str],
        court_level: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns raw LLM explanation and retrieval metadata.
        """
        ctx = self.retrieval.retrieve_full_context(
            user_query=case_facts,
            extra_filters={"section": section or "", "jurisdiction": court_level},
        )
        sections = ctx.get("sections") or []
        judgments = ctx.get("judgments") or []
        section_cites, judgment_cites = build_citation_lists(sections, judgments)

        sections_context = "\n".join(section_cites)
        judgments_context = "\n".join(judgment_cites)

        prompt = self._build_prediction_prompt(
            case_facts=case_facts,
            court_level=court_level,
            sections_context=sections_context,
            judgments_context=judgments_context,
        )
        raw = self.llm.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT_STRATEGY,
            temperature=0.2,
        )
        meta = {
            "sections": sections,
            "judgments": judgments,
            "section_citations": section_cites,
            "judgment_citations": judgment_cites,
        }
        return raw, meta

