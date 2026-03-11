"""
Standalone inference script: generate a legal draft from CLI without running the API.
Useful for testing the pipeline and vLLM connectivity.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.draft_generator import DraftGenerator


def main():
    p = argparse.ArgumentParser(description="Generate legal draft from CLI")
    p.add_argument("--document-type", type=str, default="bail_application", help="bail_application, legal_notice, affidavit, petition, agreement")
    p.add_argument("--case-facts", type=str, required=True, help="Case facts / instructions")
    p.add_argument("--court-name", type=str, default="District Court Delhi")
    p.add_argument("--client-name", type=str, default="Applicant")
    p.add_argument("--section", type=str, default="IPC 420")
    p.add_argument("--output", type=Path, default=None, help="Write draft to file")
    args = p.parse_args()

    generator = DraftGenerator()
    result = generator.generate(
        document_type=args.document_type,
        case_facts=args.case_facts,
        court_name=args.court_name,
        client_name=args.client_name,
        section=args.section,
    )
    print("=== DRAFT ===")
    print(result["draft"])
    print("\n=== VALIDATION ===")
    print(json.dumps(result["validation"], indent=2))
    print("\nSuccess:", result["success"])
    if args.output:
        args.output = Path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result["draft"], encoding="utf-8")
        print("Written to", args.output)


if __name__ == "__main__":
    main()
