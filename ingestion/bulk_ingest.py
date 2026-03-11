"""
Bulk ingestion utilities for legal datasets.

Usage examples:
  python -m ingestion.bulk_ingest --acts path/to/acts.jsonl
  python -m ingestion.bulk_ingest --judgments path/to/judgments.jsonl
  python -m ingestion.bulk_ingest --templates path/to/templates.jsonl

Input files are expected to be JSONL with one record per line.
Records are appended to the project datasets and the unified FAISS index
is rebuilt afterwards.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import DATASETS_DIR, get_settings
from app.scripts.vector_index_management import rebuild_full_index
from app.services.dataset_manager import DatasetManager


def _append_all(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Validate JSON
            obj: Dict[str, Any] = json.loads(line)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk ingest legal datasets into the Legal AI engine.")
    parser.add_argument("--acts", type=Path, default=None, help="Source JSONL file of bare act sections.")
    parser.add_argument("--judgments", type=Path, default=None, help="Source JSONL file of judgments.")
    parser.add_argument("--templates", type=Path, default=None, help="Source JSONL file of draft templates.")
    parser.add_argument("--training", type=Path, default=None, help="Source JSONL file of training examples (facts+draft).")
    args = parser.parse_args()

    settings = get_settings()
    total = 0
    manager = DatasetManager()
    if args.acts:
        total += _append_all(args.acts, DATASETS_DIR / "bare_acts.jsonl")
        manager.bump("bare_acts", source=str(args.acts))
    if args.judgments:
        total += _append_all(args.judgments, DATASETS_DIR / "judgments.jsonl")
        manager.bump("judgments", source=str(args.judgments))
    if args.templates:
        total += _append_all(args.templates, DATASETS_DIR / "draft_templates.jsonl")
        manager.bump("draft_templates", source=str(args.templates))
    if args.training:
        total += _append_all(args.training, settings.dataset_path)
        manager.bump("legal_drafts", source=str(args.training))

    print(f"Ingested {total} records. Rebuilding unified vector index...")
    count = rebuild_full_index()
    print(f"Rebuilt vector index with {count} items.")


if __name__ == "__main__":
    main()

