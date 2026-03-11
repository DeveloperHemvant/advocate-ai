from __future__ import annotations

from typing import Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.services.draft_generator import DraftGenerator
from app.utils.prompt_builder import SYSTEM_PROMPT_LEGAL


class CourtFilingService:
    """
    Wraps DraftGenerator and applies jurisdiction-specific filing instructions via LLM.
    """

    def __init__(
        self,
        draft_generator: Optional[DraftGenerator] = None,
        llm_client: Optional[LegalLLMClient] = None,
    ) -> None:
        settings = get_settings()
        self.draft_generator = draft_generator or DraftGenerator()
        self.llm = llm_client or self.draft_generator.llm  # reuse same client
        self.settings = settings

    def generate_court_document(
        self,
        document_type: str,
        case_facts: str,
        jurisdiction: str,
        extra_context: Optional[dict] = None,
    ) -> str:
        base = self.draft_generator.generate(
            document_type=document_type,
            case_facts=case_facts,
            extra_context=extra_context,
        )
        draft_text = base["draft"]
        filing_instructions = (
            f"You are preparing a court-ready filing for {jurisdiction} in India. "
            "Reformat the following draft so that it strictly follows the standard heading, "
            "cause title, prayer, verification, and annexure referencing conventions for that forum. "
            "Do not change the substantive content, only the structure and formatting.\n\n"
            f"Draft:\n{draft_text}"
        )
        formatted = self.llm.complete(
            filing_instructions,
            system_prompt=SYSTEM_PROMPT_LEGAL,
            temperature=0.2,
        )
        return formatted

