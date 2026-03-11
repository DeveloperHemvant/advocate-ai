"""
Judgment analysis service for Harvey-level Legal Intelligence.

Uses the LLM to turn a full judgment text into a structured summary
matching the JudgmentAnalysis DB model fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient
from app.utils.prompt_builder import SYSTEM_PROMPT_JUDGMENT_SUMMARY, build_judgment_summary_prompt


@dataclass
class JudgmentAnalysisResult:
  facts_summary: str
  legal_issues: str
  court_reasoning: str
  final_decision: str
  key_citations: str | None = None

  def to_dict(self) -> dict[str, str]:
    return {
        "facts_summary": self.facts_summary,
        "legal_issues": self.legal_issues,
        "court_reasoning": self.court_reasoning,
        "final_decision": self.final_decision,
        "key_citations": self.key_citations or "",
    }


class JudgmentAnalysisService:
  def __init__(self, llm: Optional[LegalLLMClient] = None) -> None:
    settings = get_settings()
    self.llm = llm or LegalLLMClient(
        base_url=settings.vllm_base_url,
        model_name=settings.llm_model_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=getattr(settings, "llm_timeout", 300.0),
    )

  def analyze_text(self, *, text: str, extra_context: str | None = None) -> JudgmentAnalysisResult:
    prompt = build_judgment_summary_prompt(judgment_text=text, extra_context=extra_context)
    raw = self.llm.complete(prompt, system_prompt=SYSTEM_PROMPT_JUDGMENT_SUMMARY, temperature=0.2)

    # Heuristic parsing into sections
    lower = raw.lower()
    def _extract(label: str) -> str:
      marker = f"{label.lower()}:"
      if marker not in lower:
        return ""
      start = lower.index(marker) + len(marker)
      # split at next heading or end
      next_markers = ["facts:", "legal issues:", "decision:", "ratio decidendi:", "citations:"]
      next_positions = [lower.find(m, start) for m in next_markers if lower.find(m, start) != -1]
      end = min(next_positions) if next_positions else len(lower)
      return raw[start:end].strip()

    facts = _extract("facts") or raw
    issues = _extract("legal issues")
    decision = _extract("decision")
    ratio = _extract("ratio decidendi")
    citations = _extract("citations")

    court_reasoning = ratio or decision

    return JudgmentAnalysisResult(
        facts_summary=facts,
        legal_issues=issues or "",
        court_reasoning=court_reasoning or "",
        final_decision=decision or "",
        key_citations=citations or "",
    )

