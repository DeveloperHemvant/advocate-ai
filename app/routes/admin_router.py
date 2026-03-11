import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import DATASETS_DIR, get_settings
from app.scripts import vector_index_management
from app.services.dataset_manager import DatasetManager


router = APIRouter(prefix="/admin", tags=["admin"])


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class BareActRecord(BaseModel):
    act_name: str = Field(..., description="Name of the Act (e.g. 'Negotiable Instruments Act, 1881').")
    section_number: str = Field(..., description="Section number (e.g. '138').")
    title: Optional[str] = None
    text: str = Field(..., description="Full text of the section.")
    keywords: Optional[List[str]] = Field(default=None, description="Optional keywords to help retrieval.")


class JudgmentRecord(BaseModel):
    case_name: str
    citation: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    facts: Optional[str] = None
    issues: Optional[str] = None
    decision: Optional[str] = None
    ratio: Optional[str] = None


class DraftTemplateRecord(BaseModel):
    name: str
    document_type: str
    template: str
    facts_example: Optional[str] = None


class TrainingExampleRecord(BaseModel):
    document_type: str
    facts: str
    draft: str


class AdminActionResponse(BaseModel):
    success: bool
    message: str


@router.post("/acts", response_model=AdminActionResponse)
def add_bare_act_section(record: BareActRecord) -> AdminActionResponse:
    """
    Add or update a bare act section in the local dataset.
    """
    path = DATASETS_DIR / "bare_acts.jsonl"
    _append_jsonl(path, record.model_dump())
    DatasetManager().bump("bare_acts", source="admin_api")
    return AdminActionResponse(success=True, message="Bare act section added.")


@router.post("/judgments", response_model=AdminActionResponse)
def add_judgment(record: JudgmentRecord) -> AdminActionResponse:
    """
    Add a judgment record to the local dataset.
    """
    path = DATASETS_DIR / "judgments.jsonl"
    _append_jsonl(path, record.model_dump())
    DatasetManager().bump("judgments", source="admin_api")
    return AdminActionResponse(success=True, message="Judgment added.")


@router.post("/draft-templates", response_model=AdminActionResponse)
def add_draft_template(record: DraftTemplateRecord) -> AdminActionResponse:
    """
    Add a reusable draft template.
    """
    path = DATASETS_DIR / "draft_templates.jsonl"
    _append_jsonl(path, record.model_dump())
    DatasetManager().bump("draft_templates", source="admin_api")
    return AdminActionResponse(success=True, message="Draft template added.")


@router.post("/training-examples", response_model=AdminActionResponse)
def add_training_example(record: TrainingExampleRecord) -> AdminActionResponse:
    """
    Add a training example (facts + draft) to the legal_drafts dataset.
    """
    settings = get_settings()
    path = settings.dataset_path
    _append_jsonl(path, record.model_dump())
    DatasetManager().bump("legal_drafts", source="admin_api")
    return AdminActionResponse(success=True, message="Training example added.")


@router.post("/rebuild-vector-index", response_model=AdminActionResponse)
def rebuild_vector_index() -> AdminActionResponse:
    """
    Trigger a full FAISS vector index rebuild from current datasets.
    """
    try:
        count = vector_index_management.rebuild_full_index()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {exc}") from exc
    return AdminActionResponse(success=True, message=f"Rebuilt vector index with {count} items.")

