"""
Orchestrates the full legal draft generation pipeline:
template selection -> RAG retrieval -> prompt build -> LLM -> validation -> formatting.
"""

import logging
from typing import Any, Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.services.rag_service import RAGService
from app.services.template_engine import fill_template, get_placeholder_keys
from app.services.validation_service import validate_draft, ValidationResult
from app.utils.formatting import extract_legal_draft_from_response
from app.utils.prompt_builder import SYSTEM_PROMPT_LEGAL, build_draft_prompt

logger = logging.getLogger(__name__)


class DraftGenerator:
    """
    End-to-end legal draft generator.
    """

    def __init__(
        self,
        llm_client: Optional[LegalLLMClient] = None,
        rag_service: Optional[RAGService] = None,
    ):
        settings = get_settings()
        self.llm = llm_client or LegalLLMClient(
            base_url=settings.vllm_base_url,
            model_name=settings.llm_model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout=getattr(settings, "llm_timeout", 300.0),
        )
        self.rag = rag_service or RAGService()

    def generate(
        self,
        document_type: str,
        case_facts: str,
        court_name: Optional[str] = None,
        client_name: Optional[str] = None,
        section: Optional[str] = None,
        extra_context: Optional[dict[str, str]] = None,
        **template_placeholders: Any,
    ) -> dict[str, Any]:
        """
        Generate a legal draft.
        Returns dict with keys: draft, validation (ValidationResult.to_dict), success.
        """
        # 1) Build placeholders for template
        placeholders = {
            "court_name": court_name or "District Court",
            "client_name": client_name or "Applicant",
            "section": section or "Applicable Section",
            "generated_facts": None,  # filled by LLM
            **{k: v for k, v in template_placeholders.items() if v is not None},
        }
        template_filled = fill_template(document_type, placeholders)

        # 2) RAG retrieval
        rag_examples = self.rag.retrieve(
            document_type=document_type,
            case_facts=case_facts,
            court_name=court_name or "",
            section=section or "",
        )

        # 3) Build prompt
        user_prompt = build_draft_prompt(
            document_type=document_type,
            template_filled=template_filled,
            case_facts=case_facts,
            rag_examples=rag_examples,
            extra_context=extra_context,
        )

        # 4) LLM call
        try:
            raw = self.llm.complete(
                user_prompt,
                system_prompt=SYSTEM_PROMPT_LEGAL,
                temperature=0.3,
            )
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            return {
                "draft": "",
                "validation": ValidationResult(False, [str(e)], []).to_dict(),
                "success": False,
            }

        # 5) Extract and format draft
        draft = extract_legal_draft_from_response(raw)

        # 6) Replace {generated_facts} in template with model output (if we kept template structure)
        # Here we return the model output as the draft; alternatively we could merge into template.
        final_draft = draft

        # 7) Validate
        validation = validate_draft(document_type, final_draft)

        return {
            "draft": final_draft,
            "validation": validation.to_dict(),
            "success": validation.valid,
        }
