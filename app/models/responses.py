from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LegalAIBaseResponse(BaseModel):
    """
    Standardized response envelope for all Legal AI endpoints.
    """

    summary: str = Field("", description="High-level summary of the answer or document.")
    legal_sections: List[str] = Field(
        default_factory=list,
        description="List of cited statutory provisions (e.g. 'NI Act 138').",
    )
    precedents: List[str] = Field(
        default_factory=list,
        description="List of cited case law references.",
    )
    analysis: str = Field(
        "",
        description="Reasoned legal analysis or explanation supporting the answer.",
    )
    draft: str = Field(
        "",
        description="Full legal draft or generated text where applicable.",
    )
    safety_flags: List[str] = Field(
        default_factory=list,
        description="Safety / guardrail warnings or notes about the AI output.",
    )


class DraftGenerationResponse(LegalAIBaseResponse):
    """
    Response model for draft generation, extending the standard envelope.
    """

    validation: dict[str, Any] = Field(
        default_factory=dict,
        description="Validation result for the generated draft.",
    )
    success: bool = Field(True, description="Whether generation and validation succeeded.")


class ResearchResponse(LegalAIBaseResponse):
    """
    Response model for legal research answers.
    """

    retrieved_sections: List[dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant bare act sections and metadata used in the answer.",
    )
    retrieved_judgments: List[dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant judgments and metadata used in the answer.",
    )


class JudgmentSummaryResponse(LegalAIBaseResponse):
    """
    Structured judgment summarization.
    """

    facts: str = ""
    issues: str = ""
    decision: str = ""
    ratio: str = ""


class CaseStrategyResponse(LegalAIBaseResponse):
    """
    Case strategy output, focused on arguments and procedure.
    """

    arguments: List[str] = Field(default_factory=list)
    procedural_steps: List[str] = Field(default_factory=list)


class DocumentAnalysisResponse(LegalAIBaseResponse):
    """
    Document risk and clause analysis.
    """

    risky_clauses: List[str] = Field(default_factory=list)
    missing_clauses: List[str] = Field(default_factory=list)
    legal_risks: List[str] = Field(default_factory=list)
    clause_types: List[str] = Field(
        default_factory=list,
        description="Detected clause type labels (e.g. termination, indemnity, confidentiality).",
    )
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Short tags describing identified risks (e.g. 'one-sided indemnity').",
    )


class ProcedureResponse(LegalAIBaseResponse):
    """
    Response for legal procedure explanations.
    """

    steps: List[str] = Field(default_factory=list)
    documents_required: List[str] = Field(default_factory=list)
    timeline: str = ""


class ArgumentsResponse(LegalAIBaseResponse):
    """
    Response for structured legal arguments.
    """

    arguments: List[str] = Field(default_factory=list)
    supporting_sections: List[str] = Field(default_factory=list)
    supporting_cases: List[str] = Field(default_factory=list)
    counterarguments: List[str] = Field(default_factory=list)

