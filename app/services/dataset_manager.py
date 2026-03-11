from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

from app.config import DATASETS_DIR


@dataclass
class DatasetVersionInfo:
    name: str
    version: int
    updated_at: str
    source: str


class DatasetManager:
    """
    Very lightweight file-based dataset versioning.
    Tracks version numbers per logical dataset.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (DATASETS_DIR / "dataset_versions.json")
        self._versions: Dict[str, DatasetVersionInfo] | None = None

    def _load(self) -> Dict[str, DatasetVersionInfo]:
        if self._versions is not None:
            return self._versions
        if not self.path.exists():
            self._versions = {}
            return self._versions
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str, DatasetVersionInfo] = {}
        for name, info in raw.items():
            out[name] = DatasetVersionInfo(
                name=name,
                version=info.get("version", 0),
                updated_at=info.get("updated_at", ""),
                source=info.get("source", ""),
            )
        self._versions = out
        return out

    def _save(self) -> None:
        if self._versions is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {name: asdict(info) for name, info in self._versions.items()}
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def bump(self, dataset_name: str, source: str) -> DatasetVersionInfo:
        versions = self._load()
        current = versions.get(dataset_name)
        version_num = (current.version + 1) if current else 1
        info = DatasetVersionInfo(
            name=dataset_name,
            version=version_num,
            updated_at=datetime.utcnow().isoformat() + "Z",
            source=source,
        )
        versions[dataset_name] = info
        self._save()
        return info

