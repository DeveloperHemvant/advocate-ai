"""
Formatting utilities for legal documents.
Cleans and structures generated text for consistent output.
"""

import re
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines and strip."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_trailing_newline(text: str) -> str:
    """Ensure document ends with a single newline."""
    t = text.strip()
    return t + "\n" if t else ""


def extract_legal_draft_from_response(raw: str) -> str:
    """
    Extract the main draft content from LLM output, removing markdown code blocks
    or obvious non-draft prefixes.
    """
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    # Remove markdown code block if present
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    # Remove common LLM preamble lines
    for prefix in (
        "Here is the draft",
        "Here's the draft",
        "Draft document:",
        "Legal draft:",
        "Generated draft:",
    ):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix) :].lstrip(":\n ")
            break
    return ensure_trailing_newline(normalize_whitespace(s))
