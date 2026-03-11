"""
Legal reasoning (IRAC-style) service for Harvey-level drafting.

This module focuses on producing a structured reasoning object that can be
stored in the legal_ai.legal_reasoning table or used inline in prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient


IRAC_SYSTEM_PROMPT = (
    "You are a senior Indian advocate. Given the case facts and any context, "
    "you must produce structured legal reasoning using the IRAC framework: "
    "Issue, Rule, Application, Conclusion. Where possible, reference relevant "
    "Indian statutory provisions and leading cases, but do not fabricate case names."
)


@dataclass
class LegalReasoningIRAC:
  """
  Simple container for IRAC reasoning used by both the API layer
  and the Node backend (via FastAPI).
  """

  issue: str
  rule: str
  application: str
  conclusion: str
  citations: str | None = None

  def to_dict(self) -> dict[str, str]:
    return {
        "issue": self.issue,
        "rule": self.rule,
        "application": self.application,
        "conclusion": self.conclusion,
        "citations": self.citations or "",
    }


class LegalReasoningService:
  """
  Light wrapper around the LLM that prompts for IRAC-style reasoning.
  """

  def __init__(self, llm: Optional[LegalLLMClient] = None) -> None:
    settings = get_settings()
    self.llm = llm or LegalLLMClient(
        base_url=settings.vllm_base_url,
        model_name=settings.llm_model_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=getattr(settings, "llm_timeout", 300.0),
    )

  def generate_irac(
      self,
      *,
      case_type: str,
      facts: str,
      context_text: str = "",
  ) -> LegalReasoningIRAC:
    """
    Generate structured IRAC reasoning from facts and optional retrieved context.
    The output is text sections that the caller may store or feed into drafting.
    """
    parts = [
        f"Case type: {case_type}",
        "",
        "## Facts",
        facts,
    ]
    if context_text:
      parts.extend(["", "## Context (authorities, statutes, prior cases)", context_text])
    parts.extend(
        [
            "",
            "Provide your reasoning strictly in the following labelled sections:",
            "Issue:",
            "Rule:",
            "Application:",
            "Conclusion:",
            "Citations: (optional list of key cases and statutes, if any)",
        ]
    )
    prompt = "\n".join(parts)
    raw = self.llm.complete(prompt, system_prompt=IRAC_SYSTEM_PROMPT, temperature=0.2)

    # Simple heuristic parsing based on headings
    lower = raw.lower()
    def _extract(label: str, fallback: str = "") -> str:
      marker = f"{label.lower()}:"
      if marker not in lower:
        return fallback
      start = lower.index(marker) + len(marker)
      # split at next heading or end
      next_markers = ["issue:", "rule:", "application:", "conclusion:", "citations:"]
      next_positions = [lower.find(m, start) for m in next_markers if lower.find(m, start) != -1]
      end = min(next_positions) if next_positions else len(lower)
      return raw[start:end].strip()

    issue = _extract("Issue", raw)
    rule = _extract("Rule", "")
    application = _extract("Application", "")
    conclusion = _extract("Conclusion", "")
    citations = _extract("Citations", "")

    return LegalReasoningIRAC(
        issue=issue or raw,
        rule=rule or "",
        application=application or "",
        conclusion=conclusion or "",
        citations=citations or "",
    )

