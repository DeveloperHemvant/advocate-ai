from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


COMMON_CLAUSE_TYPES = [
    "termination",
    "indemnity",
    "confidentiality",
    "dispute_resolution",
    "governing_law",
    "liability",
]


@dataclass
class ClauseInsight:
    text: str
    types: List[str]
    risky: bool
    risk_flags: List[str]


class ClauseIntelligenceService:
    """
    Heuristic clause intelligence layer for contracts.
    """

    def detect_clauses(self, document_text: str) -> List[ClauseInsight]:
        """
        Very lightweight rule-based clause detector over headings + paragraphs.
        """
        clauses: List[ClauseInsight] = []
        # Split into paragraphs
        paragraphs = [p.strip() for p in document_text.split("\n") if p.strip()]
        for p in paragraphs:
            lower = p.lower()
            types: List[str] = []
            risky = False
            risk_flags: List[str] = []

            if "terminate" in lower or "termination" in lower:
                types.append("termination")
            if "indemnify" in lower or "indemnity" in lower:
                types.append("indemnity")
            if "confidential" in lower or "non-disclosure" in lower:
                types.append("confidentiality")
            if "arbitration" in lower or "jurisdiction" in lower or "dispute" in lower:
                types.append("dispute_resolution")
            if "governing law" in lower or "laws of" in lower:
                types.append("governing_law")
            if "liability" in lower or "limitation of liability" in lower:
                types.append("liability")

            # Simple risk heuristics
            if "sole discretion" in lower or "without notice" in lower:
                risky = True
                risk_flags.append("one_sided_discretion")
            if "unlimited liability" in lower or "without any cap" in lower:
                risky = True
                risk_flags.append("unlimited_liability")
            if "indemnify" in lower and "negligence" not in lower:
                risky = True
                risk_flags.append("broad_indemnity")

            if types or risky:
                clauses.append(ClauseInsight(text=p, types=types, risky=risky, risk_flags=risk_flags))

        return clauses

    def infer_missing_clause_types(self, present_types: List[str]) -> List[str]:
        missing = []
        for ct in COMMON_CLAUSE_TYPES:
            if ct not in present_types:
                missing.append(ct)
        return missing

