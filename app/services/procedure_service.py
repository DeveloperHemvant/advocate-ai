from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import DATASETS_DIR, get_settings
from app.models.legal_llm import LegalLLMClient
from app.utils.prompt_builder import (
    SYSTEM_PROMPT_PROCEDURE,
    build_procedure_prompt,
)
from app.services.legal_retrieval_service import LegalRetrievalService


class ProcedureService:
    """
    Provides structured legal procedures for common issues (e.g. cheque bounce, divorce).
    """

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.dataset_path = dataset_path or (DATASETS_DIR / "legal_procedures.jsonl")
        self._cache: List[Dict[str, Any]] | None = None

    def _load(self) -> List[Dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if not self.dataset_path.exists():
            self._cache = []
            return self._cache
        out: List[Dict[str, Any]] = []
        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        self._cache = out
        return out

    def find_base_procedure(self, legal_issue: str, jurisdiction: Optional[str]) -> str:
        """
        Return a short bullet list of standard steps from the dataset, if any.
        """
        records = self._load()
        if not records:
            return ""
        q = legal_issue.lower()
        best: Optional[Dict[str, Any]] = None
        for r in records:
            issues = " ".join(r.get("issues", [])) + " " + r.get("name", "") + " " + r.get("description", "")
            if q in issues.lower():
                if jurisdiction and jurisdiction.lower() not in str(r.get("jurisdiction", "")).lower():
                    continue
                best = r
                break
        if not best:
            return ""
        lines: List[str] = []
        for step in best.get("steps", []):
            lines.append(f"- {step}")
        return "\n".join(lines)


class ProcedureEngine:
    """
    Orchestrates retrieval + LLM for procedure explanations.
    """

    def __init__(
        self,
        procedure_service: Optional[ProcedureService] = None,
        retrieval_service: Optional[LegalRetrievalService] = None,
        llm_client: Optional[LegalLLMClient] = None,
    ) -> None:
        self.procedures = procedure_service or ProcedureService()
        self.retrieval = retrieval_service or LegalRetrievalService()
        settings = get_settings()
        self.llm = llm_client or LegalLLMClient(
            base_url=settings.vllm_base_url,
            model_name=settings.llm_model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout=getattr(settings, "llm_timeout", 300.0),
        )

    def explain_procedure(
        self,
        legal_issue: str,
        jurisdiction: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns raw LLM explanation and retrieval metadata.
        """
        base_steps = self.procedures.find_base_procedure(legal_issue, jurisdiction)
        ctx = self.retrieval.retrieve_full_context(user_query=legal_issue)
        prompt = build_procedure_prompt(
            legal_issue=legal_issue,
            jurisdiction=jurisdiction,
            base_steps=base_steps,
        )
        raw = self.llm.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT_PROCEDURE,
            temperature=0.2,
        )
        return raw, ctx

