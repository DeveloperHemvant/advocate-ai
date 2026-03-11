"""
Advanced AI endpoints for Harvey-level Legal Intelligence.

These are thin HTTP wrappers around internal services: reasoning,
arguments, citations, embeddings, and judgment analysis.
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.argument_service import ArgumentService
from app.services.citation_service import build_citation_lists
from app.services.judgment_analysis_service import JudgmentAnalysisService
from app.services.legal_retrieval_service import LegalRetrievalService
from app.services.rag_service import RAGService, embed_texts, get_embedding_model
from app.services.reasoning_service import LegalReasoningIRAC, LegalReasoningService


router = APIRouter(prefix="", tags=["advanced-ai"])


class ReasoningRequest(BaseModel):
  case_type: str = Field(..., description="High-level case type e.g. bail_application, quashing_fir")
  facts: str = Field(..., description="Material facts of the case")
  context: Optional[str] = Field(
      None,
      description="Optional pre-retrieved context (citations, statutes, prior cases). "
      "If omitted, the service will rely on LLM only.",
  )


class ReasoningResponse(BaseModel):
  issue: str
  rule: str
  application: str
  conclusion: str
  citations: str = ""


@router.post("/generate-reasoning", response_model=ReasoningResponse)
def generate_reasoning(req: ReasoningRequest) -> ReasoningResponse:
  service = LegalReasoningService()
  irac: LegalReasoningIRAC = service.generate_irac(
      case_type=req.case_type,
      facts=req.facts,
      context_text=req.context or "",
  )
  return ReasoningResponse(**irac.to_dict())


class ArgumentRequest(BaseModel):
  case_type: str
  facts: str
  jurisdiction: Optional[str] = None


class ArgumentResponse(BaseModel):
  arguments: str
  supporting_sections: str
  supporting_cases: str
  counter_arguments: str


@router.post("/generate-arguments", response_model=ArgumentResponse)
def generate_arguments(req: ArgumentRequest) -> ArgumentResponse:
  service = ArgumentService()
  result = service.generate_arguments(
      case_type=req.case_type,
      facts=req.facts,
      jurisdiction=req.jurisdiction,
  )
  return ArgumentResponse(**result.to_dict())


class CitationRequest(BaseModel):
  query: str = Field(..., description="Facts or draft text to find citations for")
  top_k_sections: int = 5
  top_k_judgments: int = 5


class CitationItem(BaseModel):
  text: str


class CitationResponse(BaseModel):
  statutory: List[str]
  judgments: List[str]


@router.post("/generate-citations", response_model=CitationResponse)
def generate_citations(req: CitationRequest) -> CitationResponse:
  retrieval = LegalRetrievalService()
  ctx = retrieval.retrieve_full_context(
      user_query=req.query,
      top_k_sections=req.top_k_sections,
      top_k_judgments=req.top_k_judgments,
  )
  sections = ctx.get("sections") or []
  judgments = ctx.get("judgments") or []
  section_cites, judgment_cites = build_citation_lists(sections, judgments)
  return CitationResponse(statutory=section_cites, judgments=judgment_cites)


class EmbeddingRequest(BaseModel):
  texts: List[str] = Field(..., description="Plain text snippets to embed")


class EmbeddingResponse(BaseModel):
  vectors: List[List[float]]


@router.post("/generate-embeddings", response_model=EmbeddingResponse)
def generate_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
  settings_model = get_settings()
  model = get_embedding_model(settings_model.use_instructor)
  arr = embed_texts(model, req.texts)
  # Convert numpy array to plain list for JSON
  vectors: List[List[float]] = arr.tolist()  # type: ignore[assignment]
  return EmbeddingResponse(vectors=vectors)


class JudgmentAnalysisRequest(BaseModel):
  text: str = Field(..., description="Full text of the judgment to analyse")
  extra_context: Optional[str] = None


class JudgmentAnalysisResponse(BaseModel):
  facts_summary: str
  legal_issues: str
  court_reasoning: str
  final_decision: str
  key_citations: str = ""


@router.post("/analyze-judgment", response_model=JudgmentAnalysisResponse)
def analyze_judgment(req: JudgmentAnalysisRequest) -> JudgmentAnalysisResponse:
  service = JudgmentAnalysisService()
  result = service.analyze_text(text=req.text, extra_context=req.extra_context)
  return JudgmentAnalysisResponse(**result.to_dict())

