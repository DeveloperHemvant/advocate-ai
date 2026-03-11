from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheEntry:
    value: Any


class CacheService:
    """
    Very simple in-memory cache for retrieval contexts and LLM responses.
    """

    _instance: "CacheService" | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._store: Dict[str, CacheEntry] = {}

    @classmethod
    def instance(cls) -> "CacheService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CacheService()
        return cls._instance

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        return entry.value if entry else None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = CacheEntry(value=value)

