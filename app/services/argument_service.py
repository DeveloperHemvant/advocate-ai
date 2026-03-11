"""
Argument-building service for Harvey-level Legal Intelligence.

Builds structured arguments (grounds, supporting cases, counter-arguments)
using existing prompts and retrieval components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.services.legal_retrieval_service import LegalRetrievalService
from app.utils.prompt_builder import SYSTEM_PROMPT_ARGUMENTS, build_arguments_prompt


@dataclass
class GeneratedArguments:
  arguments: str
  supporting_sections: str
  supporting_cases: str
  counter_arguments: str

  def to_dict(self) -> dict[str, str]:
    return {
        "arguments": self.arguments,
        "supporting_sections": self.supporting_sections,
        "supporting_cases": self.supporting_cases,
        "counter_arguments": self.counter_arguments,
    }


class ArgumentService:
  """
  High-level wrapper that combines retrieval + LLM prompts
  to construct reusable argument structures.
  """

  def __init__(
      self,
      llm: Optional[LegalLLMClient] = None,
      retrieval: Optional[LegalRetrievalService] = None,
  ) -> None:
    settings = get_settings()
    self.llm = llm or LegalLLMClient(
        base_url=settings.vllm_base_url,
        model_name=settings.llm_model_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=getattr(settings, "llm_timeout", 300.0),
    )
    self.retrieval = retrieval or LegalRetrievalService()

  def generate_arguments(
      self,
      *,
      case_type: str,
      facts: str,
      jurisdiction: Optional[str] = None,
  ) -> GeneratedArguments:
    # Use existing retrieval on bare acts & judgments
    ctx = self.retrieval.retrieve_full_context(user_query=facts, document_type=None)
    sections = ctx.get("sections") or []
    judgments = ctx.get("judgments") or []

    sections_text = "\n".join(
        f"- {s.get('act_name', s.get('act', 'Act'))} section {s.get('section_number', s.get('section', ''))}: "
        f"{s.get('title', '')}"
        for s in sections
    )
    judgments_text = "\n".join(
        f"- {j.get('case_name', j.get('title', 'Unknown v. Unknown'))} ({j.get('citation', j.get('year', ''))})"
        for j in judgments
    )

    prompt = build_arguments_prompt(
        case_facts=facts,
        sections_context=sections_text,
        judgments_context=judgments_text,
        jurisdiction=jurisdiction,
    )
    raw = self.llm.complete(prompt, system_prompt=SYSTEM_PROMPT_ARGUMENTS, temperature=0.3)

    # Simple section splitting based on numbered headings
    lower = raw.lower()
    def _extract(label: str) -> str:
      marker = f"{label.lower()}."
      if marker not in lower:
        return ""
      start = lower.index(marker) + len(marker)
      # next numeric heading or end
      next_positions = [
          lower.find("1.", start),
          lower.find("2.", start),
          lower.find("3.", start),
          lower.find("4.", start),
      ]
      next_positions = [p for p in next_positions if p != -1]
      end = min(next_positions) if next_positions else len(lower)
      return raw[start:end].strip()

    arguments = _extract("1") or raw
    supporting_sections = _extract("2")
    supporting_cases = _extract("3")
    counter_arguments = _extract("4")

    return GeneratedArguments(
        arguments=arguments,
        supporting_sections=supporting_sections,
        supporting_cases=supporting_cases,
        counter_arguments=counter_arguments,
    )

