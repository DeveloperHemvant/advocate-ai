"""
Simple evaluation harness for the Legal AI platform.
Runs stored prompts through the API services and compares with expected outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.config import DATASETS_DIR
from app.routes.generate_draft import generate_draft, GenerateDraftRequest
from app.routes.research_router import legal_research, LegalResearchRequest


@dataclass
class EvalCase:
    task: str
    prompt: Dict[str, Any]
    expected_keywords: List[str]


class LegalAIEvaluator:
    """
    Evaluation utility for draft quality, citation accuracy, hallucination rate, etc.
    """

    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or (DATASETS_DIR / "eval_cases.jsonl")

    def _load_cases(self) -> List[EvalCase]:
        if not self.dataset_path.exists():
            return []
        out: List[EvalCase] = []
        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append(
                    EvalCase(
                        task=obj.get("task", "draft"),
                        prompt=obj.get("prompt", {}),
                        expected_keywords=obj.get("expected_keywords", []),
                    )
                )
        return out

    def run(self) -> Dict[str, Any]:
        cases = self._load_cases()
        results: List[Dict[str, Any]] = []
        for case in cases:
            if case.task == "draft":
                req = GenerateDraftRequest(**case.prompt)
                resp = generate_draft(req)
                text = resp.draft
                citations = resp.precedents
            elif case.task == "research":
                req = LegalResearchRequest(**case.prompt)
                resp = legal_research(req)
                text = resp.analysis
                citations = resp.precedents
            else:
                continue
            keywords_ok = all(k.lower() in text.lower() for k in case.expected_keywords)
            hallucination_suspect = bool(resp.safety_flags)  # type: ignore[attr-defined]
            results.append(
                {
                    "task": case.task,
                    "keywords_ok": keywords_ok,
                    "num_citations": len(citations),
                    "hallucination_suspect": hallucination_suspect,
                }
            )
        return {"cases_evaluated": len(results), "results": results}

