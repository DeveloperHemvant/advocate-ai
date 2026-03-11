from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DATASETS_DIR

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    timestamp: float
    endpoint: str
    user_ip: str
    intent: Optional[str]
    latency_ms: float
    retrieval_ms: Optional[float]
    llm_ms: Optional[float]
    extra: Dict[str, Any]


class TelemetryService:
    """
    Appends telemetry records to a JSONL file for later analysis.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (DATASETS_DIR / "telemetry.jsonl")

    def log(
        self,
        *,
        endpoint: str,
        user_ip: str,
        intent: Optional[str],
        latency_ms: float,
        retrieval_ms: Optional[float] = None,
        llm_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = TelemetryRecord(
            timestamp=time.time(),
            endpoint=endpoint,
            user_ip=user_ip,
            intent=intent,
            latency_ms=latency_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            extra=extra or {},
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write telemetry: %s", exc)

