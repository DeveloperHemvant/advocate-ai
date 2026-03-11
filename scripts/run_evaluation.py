"""
CLI entry point to run the Legal AI evaluation suite.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.legal_ai_evaluator import LegalAIEvaluator  # noqa: E402


if __name__ == "__main__":
    evaluator = LegalAIEvaluator()
    summary = evaluator.run()
    import json

    print(json.dumps(summary, indent=2))

