"""
Build FAISS vector database from legal_drafts.jsonl.
Embeds drafts (or facts+draft) using BGE-small / Instructor and saves index + metadata.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.rag_service import get_embedding_model, embed_texts
from app.vectorstore.faiss_index import FAISSLegalIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(
    dataset_path: Path | None = None,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
    batch_size: int = 32,
    use_instructor: bool = False,
) -> int:
    """
    Load JSONL, embed each (facts + draft) for retrieval, add to FAISS, save.
    Returns number of documents indexed.
    """
    settings = get_settings()
    dataset_path = dataset_path or settings.dataset_path
    index_path = index_path or settings.vector_index_path
    metadata_path = metadata_path or settings.vector_metadata_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Create legal_drafts.jsonl first.")

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("Dataset is empty")

    logger.info("Loading embedding model (use_instructor=%s)...", use_instructor or settings.use_instructor)
    model = get_embedding_model(use_instructor or settings.use_instructor)
    dim = settings.embedding_dim

    # Text to embed: concatenation of document_type, facts, and draft for better retrieval
    texts = []
    metadata = []
    for r in records:
        text = f"Document type: {r.get('document_type', '')}. Facts: {r.get('facts', '')}. Draft: {r.get('draft', '')}"
        texts.append(text[:8000])  # cap length for embedding
        metadata.append({
            "document_type": r.get("document_type", ""),
            "facts": r.get("facts", ""),
            "draft": r.get("draft", ""),
        })

    index = FAISSLegalIndex(dimension=dim, index_path=index_path, metadata_path=metadata_path)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_meta = metadata[i: i + batch_size]
        emb = embed_texts(
            model,
            batch_texts,
            instruction="Represent the legal document for retrieval: " if (use_instructor or settings.use_instructor) else None,
        )
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        index.add(emb, batch_meta)
        logger.info("Indexed %d / %d", min(i + batch_size, len(texts)), len(texts))

    index.save(index_path, metadata_path)
    logger.info("Saved FAISS index (%d vectors) to %s", index.size, index_path)
    return index.size


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build FAISS vector DB from legal_drafts.jsonl")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--instructor", action="store_true", help="Use Instructor-xl embeddings")
    args = p.parse_args()
    build_index(
        dataset_path=args.dataset,
        index_path=args.index,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
        use_instructor=args.instructor,
    )
