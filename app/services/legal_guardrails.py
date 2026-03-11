from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import DATASETS_DIR


@dataclass
class GuardrailResult:
    is_safe: bool
    missing_sections: List[str]
    missing_cases: List[str]
    warnings: List[str]


class LegalGuardrails:
    """
    Validates AI outputs against local datasets to reduce hallucinations.
    """

    def __init__(
        self,
        acts_path: Optional[Path] = None,
        judgments_path: Optional[Path] = None,
    ) -> None:
        self.acts_path = acts_path or (DATASETS_DIR / "bare_acts.jsonl")
        self.judgments_path = judgments_path or (DATASETS_DIR / "judgments.jsonl")
        self._sections_index: Dict[str, Dict[str, Any]] | None = None
        self._judgments_index: Dict[str, Dict[str, Any]] | None = None

    def _load_sections_index(self) -> Dict[str, Dict[str, Any]]:
        if self._sections_index is not None:
            return self._sections_index
        out: Dict[str, Dict[str, Any]] = {}
        if self.acts_path.exists():
            with self.acts_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    act = rec.get("act_name") or rec.get("act") or ""
                    num = rec.get("section_number") or rec.get("section") or ""
                    key = f"{act}:{num}".lower()
                    out[key] = rec
        self._sections_index = out
        return out

    def _load_judgments_index(self) -> Dict[str, Dict[str, Any]]:
        if self._judgments_index is not None:
            return self._judgments_index
        out: Dict[str, Dict[str, Any]] = {}
        if self.judgments_path.exists():
            with self.judgments_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    name = rec.get("case_name") or rec.get("title") or ""
                    citation = rec.get("citation") or ""
                    if citation:
                        out[citation.lower()] = rec
                    if name:
                        out[name.lower()] = rec
        self._judgments_index = out
        return out

    def check_citations(
        self,
        legal_sections: List[str],
        precedents: List[str],
    ) -> GuardrailResult:
        sections_idx = self._load_sections_index()
        judgments_idx = self._load_judgments_index()

        missing_sections: List[str] = []
        missing_cases: List[str] = []

        for sec in legal_sections:
            key = sec.lower().replace("section", "").strip()
            found = False
            for skey in sections_idx.keys():
                if key in skey:
                    found = True
                    break
            if not found:
                missing_sections.append(sec)

        for case in precedents:
            key = case.lower()
            if key not in judgments_idx:
                # Try more relaxed matching
                matched = any(key in jkey for jkey in judgments_idx.keys())
                if not matched:
                    missing_cases.append(case)

        warnings: List[str] = []
        if missing_sections:
            warnings.append(f"Some sections not found in local dataset: {', '.join(missing_sections[:5])}")
        if missing_cases:
            warnings.append(f"Some case citations not found in local dataset: {', '.join(missing_cases[:5])}")

        is_safe = not (missing_sections or missing_cases)
        return GuardrailResult(
            is_safe=is_safe,
            missing_sections=missing_sections,
            missing_cases=missing_cases,
            warnings=warnings,
        )

