from __future__ import annotations

import io
import logging
from typing import Any, Tuple

from fastapi import UploadFile

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.services.clause_intelligence_service import ClauseIntelligenceService
from app.services.legal_retrieval_service import LegalRetrievalService
from app.utils.prompt_builder import (
    SYSTEM_PROMPT_DOCUMENT_ANALYSIS,
    build_document_analysis_prompt,
)

logger = logging.getLogger(__name__)


def _extract_text_from_pdf(data: bytes) -> str:
    try:
        import pdfplumber  # type: ignore[import]
    except ImportError:
        logger.warning("pdfplumber not installed; returning raw bytes decoded as UTF-8 where possible.")
        return data.decode("utf-8", errors="ignore")
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _extract_text_from_docx(data: bytes) -> str:
    try:
        import docx  # type: ignore[import]
    except ImportError:
        logger.warning("python-docx not installed; returning raw bytes decoded as UTF-8 where possible.")
        return data.decode("utf-8", errors="ignore")
    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_upload(file: UploadFile) -> str:
    """
    Extract text from PDF, DOCX, or TXT upload.
    """
    data = file.file.read()
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()

    if content_type == "application/pdf" or filename.endswith(".pdf"):
        return _extract_text_from_pdf(data)
    if (
        content_type
        in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }
        or filename.endswith(".docx")
    ):
        return _extract_text_from_docx(data)

    # Fallback: treat as plain text
    return data.decode("utf-8", errors="ignore")


class DocumentAnalyzer:
    """
    High-level document analysis pipeline:
    extract text -> retrieve legal context -> LLM analysis.
    """

    def __init__(
        self,
        llm_client: LegalLLMClient | None = None,
        retrieval_service: LegalRetrievalService | None = None,
    ) -> None:
        settings = get_settings()
        self.llm = llm_client or LegalLLMClient(
            base_url=settings.vllm_base_url,
            model_name=settings.llm_model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout=getattr(settings, "llm_timeout", 300.0),
        )
        self.retrieval = retrieval_service or LegalRetrievalService()
        self.clause_intel = ClauseIntelligenceService()

    def analyze(self, document_text: str) -> Tuple[str, dict[str, Any], dict[str, Any]]:
        """
        Run full analysis and return:
        - raw LLM output
        - retrieval metadata
        - clause intelligence summary
        """
        ctx = self.retrieval.retrieve_full_context(user_query=document_text)
        prompt = build_document_analysis_prompt(
            document_text=document_text,
            context_text=ctx.get("context_text", ""),
        )
        raw = self.llm.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT_DOCUMENT_ANALYSIS,
            temperature=0.2,
        )

        insights = self.clause_intel.detect_clauses(document_text)
        present_types: list[str] = []
        risk_flags: list[str] = []
        risky_clauses: list[str] = []
        for ins in insights:
            present_types.extend(ins.types)
            if ins.risky:
                risky_clauses.append(ins.text)
                risk_flags.extend(ins.risk_flags)
        missing_types = self.clause_intel.infer_missing_clause_types(sorted(set(present_types)))

        clause_summary = {
            "present_types": sorted(set(present_types)),
            "missing_types": missing_types,
            "risky_clauses": risky_clauses,
            "risk_flags": sorted(set(risk_flags)),
        }
        return raw, ctx, clause_summary


