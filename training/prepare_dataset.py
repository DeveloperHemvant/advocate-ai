"""
Prepare legal drafts dataset for training and vector DB.
Validates JSONL schema, normalizes document_type, and optionally splits train/val.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.template_engine import DOCUMENT_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"document_type", "facts", "draft"}


def validate_line(line: str, line_no: int) -> tuple[bool, dict | None, str | None]:
    """
    Validate a single JSONL line. Returns (ok, record, error_message).
    """
    line = line.strip()
    if not line:
        return False, None, "Empty line"
    try:
        record = json.loads(line)
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"
    if not isinstance(record, dict):
        return False, None, "Record must be a JSON object"
    missing = REQUIRED_KEYS - set(record.keys())
    if missing:
        return False, None, f"Missing keys: {missing}"
    if record.get("document_type") not in DOCUMENT_TYPES:
        return False, None, f"document_type must be one of {DOCUMENT_TYPES}"
    if not (record.get("facts") and record.get("draft")):
        return False, None, "facts and draft must be non-empty"
    return True, record, None


def prepare_dataset(
    input_path: Path,
    output_path: Path | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Read JSONL from input_path, validate, optionally write cleaned output and train/val split.
    Returns (total_count, error_count).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    records = []
    errors = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            ok, record, err = validate_line(line, i)
            if ok:
                records.append(record)
            else:
                errors += 1
                logger.warning("Line %d: %s", i, err)

    logger.info("Valid records: %d, errors: %d", len(records), errors)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote cleaned dataset to %s", output_path)

    if val_ratio > 0 and records:
        import random
        random.seed(seed)
        random.shuffle(records)
        n_val = max(1, int(len(records) * val_ratio))
        val_records = records[:n_val]
        train_records = records[n_val:]
        train_path = input_path.parent / (input_path.stem + "_train.jsonl")
        val_path = input_path.parent / (input_path.stem + "_val.jsonl")
        for path, data in [(train_path, train_records), (val_path, val_records)]:
            with open(path, "w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Train: %s (%d), Val: %s (%d)", train_path, len(train_records), val_path, len(val_records))

    return len(records), errors


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare legal drafts JSONL dataset")
    parser.add_argument("input", type=Path, default=PROJECT_ROOT / "datasets" / "legal_drafts.jsonl", nargs="?")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Cleaned output path")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.input, args.output, args.val_ratio, args.seed)
