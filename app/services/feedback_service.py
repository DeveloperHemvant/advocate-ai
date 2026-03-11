from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DATASETS_DIR


@dataclass
class FeedbackRecord:
    original_prompt: str
    ai_output: str
    user_corrected_output: Optional[str]
    rating: Optional[int]
    task_type: Optional[str]
    metadata: Dict[str, Any]


class FeedbackService:
    """
    Persists user feedback for continuous learning / LoRA fine-tuning.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (DATASETS_DIR / "ai_feedback.jsonl")

    def save_feedback(
        self,
        original_prompt: str,
        ai_output: str,
        user_corrected_output: Optional[str],
        rating: Optional[int],
        task_type: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = FeedbackRecord(
            original_prompt=original_prompt,
            ai_output=ai_output,
            user_corrected_output=user_corrected_output,
            rating=rating,
            task_type=task_type,
            metadata=metadata or {},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

