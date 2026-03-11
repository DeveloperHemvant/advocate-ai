"""
Citation helper utilities.
Formats statutory and case law citations for use by LLM prompts and responses.
"""

from typing import Any, Iterable, List


def format_section_citation(section: dict[str, Any]) -> str:
    """
    Format a single bare act section into a human-readable citation string.
    Expected keys: act_name, section_number, title (optional).
    """
    act = section.get("act_name") or section.get("act") or "Unknown Act"
    num = section.get("section_number") or section.get("section") or ""
    title = section.get("title") or ""
    base = f"Section {num} of the {act}" if num else act
    if title:
        return f"{base} ({title})"
    return base


def format_case_citation(judgment: dict[str, Any]) -> str:
    """
    Format a single judgment record into a standard case citation string.
    Expected keys: case_name, citation, year, court (all optional).
    """
    case_name = judgment.get("case_name") or judgment.get("title") or "Unknown v. Unknown"
    citation = judgment.get("citation") or ""
    year = judgment.get("year") or ""
    court = judgment.get("court") or ""

    parts: List[str] = [case_name]
    if citation:
        parts.append(f"({citation})")
    elif year:
        parts.append(f"({year})")
    if court:
        parts.append(court)
    return " ".join(parts).strip()


def build_citation_lists(
    sections: Iterable[dict[str, Any]] | None,
    judgments: Iterable[dict[str, Any]] | None,
) -> tuple[List[str], List[str]]:
    """
    Convert structured sections and judgments into lists of citation strings.
    """
    section_cites: List[str] = []
    judgment_cites: List[str] = []
    if sections:
        section_cites = [format_section_citation(s) for s in sections]
    if judgments:
        judgment_cites = [format_case_citation(j) for j in judgments]
    return section_cites, judgment_cites

