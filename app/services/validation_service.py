"""
Validation service for generated legal documents.
Checks structure and required sections for each document type.
"""

import re
from typing import Any

from app.services.template_engine import DOCUMENT_TYPES


# Required phrases/sections per document type (Indian legal conventions)
REQUIRED_STRUCTURE = {
    "bail_application": [
        "court",
        "bail",
        "applicant",
        "prayer",
        "hon'ble",
    ],
    "legal_notice": [
        "legal notice",
        "section 80",
        "code of civil procedure",
        "date",
    ],
    "affidavit": [
        "affidavit",
        "solemnly",
        "verify",
        "deponent",
        "verification",
    ],
    "petition": [
        "court",
        "petition",
        "petitioner",
        "respondent",
        "prayer",
    ],
    "agreement": [
        "agreement",
        "between",
        "whereas",
        "witness",
    ],
}


class ValidationResult:
    """Result of document validation."""

    def __init__(self, valid: bool, errors: list[str], warnings: list[str]):
        self.valid = valid
        self.errors = errors
        self.warnings = warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_draft(document_type: str, draft_text: str) -> ValidationResult:
    """
    Validate generated draft for required structure and minimum content.
    Returns ValidationResult with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not draft_text or not draft_text.strip():
        return ValidationResult(False, ["Draft is empty"], [])

    if document_type not in DOCUMENT_TYPES:
        return ValidationResult(False, [f"Unknown document_type: {document_type}"], [])

    text_lower = draft_text.lower()

    # Check required structural keywords
    required = REQUIRED_STRUCTURE.get(document_type, [])
    for phrase in required:
        if phrase not in text_lower:
            warnings.append(f"Expected phrase/section not found: '{phrase}'")

    # Minimum length heuristic
    if len(draft_text.strip()) < 200:
        warnings.append("Draft is very short; consider adding more substance.")

    # Check for placeholder leakage
    if "[FACTS AND SUBSTANCE TO BE GENERATED" in draft_text or "{generated_facts}" in draft_text:
        errors.append("Template placeholder 'generated_facts' was not replaced.")

    # Check for common placeholders left unfilled
    unfilled = re.findall(r"\{(\w+)\}", draft_text)
    if unfilled:
        errors.append(f"Unfilled placeholders found: {unfilled}")

    valid = len(errors) == 0
    return ValidationResult(valid, errors, warnings)
