"""
Vector index management helpers used by the admin API and CLI scripts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import DATASETS_DIR, get_settings
from app.services.rag_service import embed_texts, get_embedding_model
from app.vectorstore.faiss_index import FAISSLegalIndex


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _build_text_for_record(kind: str, rec: Dict[str, Any]) -> str:
    if kind == "draft":
        return f"Document type: {rec.get('document_type', '')}. Facts: {rec.get('facts', '')}. Draft: {rec.get('draft', '')}"
    if kind == "bare_act":
        return (
            f"Act: {rec.get('act_name', rec.get('act', ''))}, "
            f"Section {rec.get('section_number', rec.get('section', ''))}: "
            f"{rec.get('title', '')}. Text: {rec.get('text', '')}"
        )
    if kind == "judgment":
        return (
            f"Case: {rec.get('case_name', '')} ({rec.get('citation', '')}) {rec.get('court', '')}. "
            f"Facts: {rec.get('facts', '')}. Issues: {rec.get('issues', '')}. Ratio: {rec.get('ratio', '')}"
        )
    if kind == "template":
        return (
            f"Template {rec.get('name', '')} for {rec.get('document_type', '')}. "
            f"Example facts: {rec.get('facts_example', '')}. Body: {rec.get('template', '')}"
        )
    return json.dumps(rec, ensure_ascii=False)


def rebuild_full_index() -> int:
    """
    Rebuild a single FAISS index that contains drafts, bare act sections, judgments, and templates.
    """
    settings = get_settings()
    drafts_path = settings.dataset_path
    acts_path = DATASETS_DIR / "bare_acts.jsonl"
    judgments_path = DATASETS_DIR / "judgments.jsonl"
    templates_path = DATASETS_DIR / "draft_templates.jsonl"

    drafts = _load_jsonl(drafts_path)
    acts = _load_jsonl(acts_path)
    judgments = _load_jsonl(judgments_path)
    templates = _load_jsonl(templates_path)

    all_records: List[Dict[str, Any]] = []
    texts: List[str] = []

    for r in drafts:
        r2 = dict(r)
        r2["kind"] = "draft"
        all_records.append(r2)
        texts.append(_build_text_for_record("draft", r2))
    for r in acts:
        r2 = dict(r)
        r2["kind"] = "bare_act"
        all_records.append(r2)
        texts.append(_build_text_for_record("bare_act", r2))
    for r in judgments:
        r2 = dict(r)
        r2["kind"] = "judgment"
        all_records.append(r2)
        texts.append(_build_text_for_record("judgment", r2))
    for r in templates:
        r2 = dict(r)
        r2["kind"] = "template"
        all_records.append(r2)
        texts.append(_build_text_for_record("template", r2))

    if not all_records:
        return 0

    model = get_embedding_model(get_settings().use_instructor)
    embeddings = embed_texts(model, texts)
    if embeddings.ndim == 1:
        import numpy as np

        embeddings = embeddings.reshape(1, -1)

    index = FAISSLegalIndex(
        dimension=settings.embedding_dim,
        index_path=settings.vector_index_path,
        metadata_path=settings.vector_metadata_path,
    )
    index.add(embeddings, all_records)
    index.save(settings.vector_index_path, settings.vector_metadata_path)
    return index.size

