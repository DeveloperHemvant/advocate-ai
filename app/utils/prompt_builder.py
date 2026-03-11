"""
Prompt construction for legal draft generation.
Combines template, RAG examples, and case facts into a single LLM prompt.
"""

from typing import Any


SYSTEM_PROMPT_LEGAL = """You are an expert Indian legal draftsman. Your task is to generate precise, court-ready legal documents in English for Indian courts. Follow the given template structure strictly. Use formal legal language and appropriate Indian legal terminology. Do not add disclaimers or notes—output only the draft document."""

SYSTEM_PROMPT_RESEARCH = """You are an expert Indian legal research assistant.
Answer using Indian statutes and case law only.
Always cite relevant sections and leading judgments where applicable.
Be concise, structured, and neutral."""

SYSTEM_PROMPT_JUDGMENT_SUMMARY = """You are an expert Indian legal analyst.
Summarize judgments into: (1) material facts, (2) legal issues, (3) decision, and (4) ratio decidendi.
Keep the language clear and neutral, and retain important citations."""

SYSTEM_PROMPT_STRATEGY = """You are a senior Indian advocate.
Based on the given facts and law, suggest possible arguments, counter-arguments, and procedural steps.
Do not give business or moral advice; focus only on legal strategy within Indian law."""

SYSTEM_PROMPT_DOCUMENT_ANALYSIS = """You are an expert Indian contracts and procedure lawyer.
Analyse the given document for legally risky clauses, missing protections, and compliance issues under Indian law.
Be specific and practical."""

SYSTEM_PROMPT_PROCEDURE = """You are an expert Indian procedural law assistant.
Describe legal procedures step by step, including required documents and typical timelines.
Do not provide business or financial advice."""

SYSTEM_PROMPT_ARGUMENTS = """You are a senior Indian advocate.
Draft well-structured legal arguments supported by statutory provisions and case law.
Always connect each argument to specific sections and judgments where possible."""

SYSTEM_PROMPT_TRANSLATION = """You are a precise legal translator for Indian law documents.
Translate the text faithfully while keeping legal meaning and terminology intact."""


def build_draft_prompt(
    document_type: str,
    template_filled: str,
    case_facts: str,
    rag_examples: list[dict[str, Any]],
    extra_context: dict[str, str] | None = None,
) -> str:
    """
    Build the user prompt for draft generation.
    template_filled: template with placeholders already filled (court_name, client_name, etc.)
    rag_examples: list of {"draft": "...", "facts": "...", "document_type": "..."}
    """
    parts = [
        "Generate a complete legal draft as per the following.",
        "",
        "## Template and context",
        template_filled,
        "",
        "## Case facts / instructions",
        case_facts,
    ]
    if extra_context:
        parts.append("")
        parts.append("## Additional context")
        for k, v in extra_context.items():
            if v:
                parts.append(f"- **{k}**: {v}")
    if rag_examples:
        parts.append("")
        parts.append("## Similar reference drafts (use only for style and structure)")
        for i, ex in enumerate(rag_examples[:5], 1):
            facts = ex.get("facts", "")
            draft = ex.get("draft", "")
            if facts:
                parts.append(f"Reference {i} - Facts: {facts[:300]}{'...' if len(facts) > 300 else ''}")
            if draft:
                parts.append(f"Reference {i} - Draft excerpt:\n{draft[:1200]}{'...' if len(draft) > 1200 else ''}")
            parts.append("")
    parts.append("")
    parts.append("Output the complete draft only, with no preamble or explanation.")
    return "\n".join(parts)


def build_search_query(document_type: str, case_facts: str, **kwargs: str) -> str:
    """Build a query string for RAG retrieval (embedding)."""
    parts = [f"Document type: {document_type}.", f"Facts: {case_facts}"]
    for key, value in kwargs.items():
        if value:
            parts.append(f"{key}: {value}")
    return " ".join(parts)


def build_research_prompt(
    query: str,
    *,
    context_text: str,
) -> str:
    """
    Prompt for general legal research answers.
    """
    parts = [
        "Answer the following Indian law research question using the provided context.",
        "",
        "## User query",
        query,
        "",
        "## Retrieved legal context",
        context_text or "[No additional context available]",
        "",
        "Provide a structured answer with headings, and explicitly mention statutory provisions and key cases you rely on.",
    ]
    return "\n".join(parts)


def build_judgment_summary_prompt(
    judgment_text: str,
    *,
    extra_context: str | None = None,
) -> str:
    parts = [
        "Summarize the following Indian court judgment.",
        "",
        "## Judgment text",
        judgment_text,
    ]
    if extra_context:
        parts.extend(["", "## Additional context", extra_context])
    parts.append("")
    parts.append(
        "Structure the summary into: Facts, Legal Issues, Decision, and Ratio Decidendi. "
        "Use clear numbered or bulleted lists.",
    )
    return "\n".join(parts)


def build_case_strategy_prompt(
    case_facts: str,
    *,
    sections_context: str,
    judgments_context: str,
    jurisdiction: str | None = None,
) -> str:
    parts = [
        "Provide a detailed case strategy under Indian law.",
        "",
        "## Case facts",
        case_facts,
    ]
    if jurisdiction:
        parts.extend(["", f"## Jurisdiction\n{jurisdiction}"])
    if sections_context:
        parts.extend(["", "## Relevant statutory provisions", sections_context])
    if judgments_context:
        parts.extend(["", "## Relevant judgments", judgments_context])
    parts.extend(
        [
            "",
            "Lay out:",
            "1. Possible arguments for the client.",
            "2. Likely counter-arguments.",
            "3. Supporting statutory provisions and case law.",
            "4. Recommended procedural steps.",
        ]
    )
    return "\n".join(parts)


def build_document_analysis_prompt(
    document_text: str,
    *,
    context_text: str | None = None,
) -> str:
    parts = [
        "Analyse the following legal document under Indian law.",
        "",
        "## Document text",
        document_text,
    ]
    if context_text:
        parts.extend(["", "## Additional context", context_text])
    parts.extend(
        [
            "",
            "Identify:",
            "- Risky or one-sided clauses (explain why).",
            "- Important missing clauses or protections.",
            "- Overall legal risks for the client.",
            "Provide a short summary at the end.",
        ]
    )
    return "\n".join(parts)


def build_procedure_prompt(
    legal_issue: str,
    jurisdiction: str | None,
    base_steps: str,
) -> str:
    parts = [
        "Explain the legal procedure in India for the following issue.",
        "",
        "## Legal issue",
        legal_issue,
    ]
    if jurisdiction:
        parts.extend(["", "## Jurisdiction", jurisdiction])
    if base_steps:
        parts.extend(["", "## Known standard steps", base_steps])
    parts.extend(
        [
            "",
            "Provide:",
            "- A numbered list of procedural steps.",
            "- A list of documents typically required.",
            "- An indicative timeline (where possible).",
        ]
    )
    return "\n".join(parts)


def build_arguments_prompt(
    case_facts: str,
    *,
    sections_context: str,
    judgments_context: str,
    jurisdiction: str | None = None,
) -> str:
    parts = [
        "Prepare structured legal arguments under Indian law.",
        "",
        "## Case facts",
        case_facts,
    ]
    if jurisdiction:
        parts.extend(["", "## Jurisdiction", jurisdiction])
    if sections_context:
        parts.extend(["", "## Statutory provisions", sections_context])
    if judgments_context:
        parts.extend(["", "## Case law", judgments_context])
    parts.extend(
        [
            "",
            "Output four sections:",
            "1. Arguments (for the client).",
            "2. Supporting sections (mapping arguments to statutory provisions).",
            "3. Supporting cases (mapping arguments to judgments).",
            "4. Counterarguments (likely points from the other side).",
        ]
    )
    return "\n".join(parts)


def build_translation_prompt(
    text: str,
    *,
    source_lang: str,
    target_lang: str,
) -> str:
    return (
        f"Translate the following legal text from {source_lang} to {target_lang}.\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Return only the translated text."
    )


