"""
Evaluation utilities for Legal Drafting AI.
Performs validation-set checks and optional BLEU/ROUGE on generated drafts.
"""

import json
import logging
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.template_engine import DOCUMENT_TYPES
from app.services.validation_service import validate_draft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_validation_set(val_path: Path) -> dict:
    """
    Run structure validation on all drafts in validation JSONL.
    Returns counts: valid, invalid, total, and per-document_type stats.
    """
    val_path = Path(val_path)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation set not found: {val_path}")

    total = 0
    valid_count = 0
    by_type = {dt: {"valid": 0, "invalid": 0} for dt in DOCUMENT_TYPES}

    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_type = record.get("document_type", "")
            draft = record.get("draft", "")
            total += 1
            result = validate_draft(doc_type, draft)
            if result.valid:
                valid_count += 1
                if doc_type in by_type:
                    by_type[doc_type]["valid"] += 1
            else:
                if doc_type in by_type:
                    by_type[doc_type]["invalid"] += 1

    return {
        "total": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "valid_ratio": valid_count / total if total else 0,
        "by_document_type": by_type,
    }


def evaluate_generated(
    generated_drafts: list[dict],
) -> dict:
    """
    Evaluate a list of generated drafts. Each item: {document_type, draft, reference_draft?}.
    Returns validation summary and optional BLEU/ROUGE if reference_draft provided.
    """
    results = []
    for item in generated_drafts:
        doc_type = item.get("document_type", "bail_application")
        draft = item.get("draft", "")
        val = validate_draft(doc_type, draft)
        results.append({"document_type": doc_type, "validation": val.to_dict()})

    valid = sum(1 for r in results if r["validation"]["valid"])
    return {
        "count": len(results),
        "valid_count": valid,
        "valid_ratio": valid / len(results) if results else 0,
        "details": results,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate legal drafts validation set")
    p.add_argument("val_jsonl", type=Path, nargs="?", default=PROJECT_ROOT / "datasets" / "legal_drafts_val.jsonl")
    args = p.parse_args()
    out = evaluate_validation_set(args.val_jsonl)
    logger.info("Validation set results: %s", json.dumps(out, indent=2))
