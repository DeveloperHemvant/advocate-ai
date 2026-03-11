from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.responses import DocumentAnalysisResponse
from app.services.document_analysis_service import DocumentAnalyzer, extract_text_from_upload


router = APIRouter(prefix="", tags=["documents"])


@router.post("/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)) -> DocumentAnalysisResponse:
    """
    Analyse an uploaded legal document (PDF/DOCX/TXT) for risks and clauses.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    text = extract_text_from_upload(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded document.")

    analyzer = DocumentAnalyzer()
    answer, ctx, clause_summary = analyzer.analyze(text)

    # Heuristic parsing of general risk / summary from the LLM answer
    risky_from_llm: List[str] = []
    missing_from_llm: List[str] = []
    risks_from_llm: List[str] = []
    current = None
    for line in answer.splitlines():
        lower = line.lower().strip(" :#")
        if "risky clause" in lower or "risky clauses" in lower:
            current = "risky"
            continue
        if "missing clause" in lower or "missing clauses" in lower:
            current = "missing"
            continue
        if "legal risk" in lower or "risks" in lower:
            current = "risks"
            continue
        if line.strip().startswith(("-", "*")) or line.strip()[:2].isdigit():
            content = line.strip(" -*\t")
            if current == "risky":
                risky_from_llm.append(content)
            elif current == "missing":
                missing_from_llm.append(content)
            elif current == "risks":
                risks_from_llm.append(content)

    summary = answer.splitlines()[0] if answer.strip() else ""

    combined_risky = list({*risky_from_llm, *clause_summary.get("risky_clauses", [])})

    return DocumentAnalysisResponse(
        summary=summary,
        legal_sections=[],
        precedents=[],
        analysis=answer,
        draft="",
        risky_clauses=combined_risky,
        missing_clauses=clause_summary.get("missing_types", []) or missing_from_llm,
        legal_risks=risks_from_llm,
        clause_types=clause_summary.get("present_types", []),
        risk_flags=clause_summary.get("risk_flags", []),
    )

