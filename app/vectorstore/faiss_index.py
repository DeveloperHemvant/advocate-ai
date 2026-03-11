"""
FAISS-based vector store for legal document retrieval.
Stores embeddings of legal drafts and supports semantic search for RAG.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _import_faiss():
    try:
        import faiss
        return faiss
    except ImportError as e:
        raise ImportError(
            "faiss-cpu is required. Install with: pip install faiss-cpu"
        ) from e


class FAISSLegalIndex:
    """
    FAISS index for legal draft embeddings.
    Persists index and metadata to disk for reuse.
    """

    def __init__(
        self,
        dimension: int,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._index = None
        self._metadata: list[dict[str, Any]] = []

    def _get_index(self):
        faiss = _import_faiss()
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine if normalized)
        return self._index

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add embeddings and corresponding metadata to the index."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        # Normalize for cosine similarity via inner product
        faiss = _import_faiss()
        faiss.normalize_L2(embeddings)
        index = self._get_index()
        index.add(embeddings)
        self._metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for nearest neighbours. Returns list of dicts with keys:
        draft, document_type, facts, score (and any other metadata).
        """
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss = _import_faiss()
        faiss.normalize_L2(query_embedding)
        index = self._get_index()
        scores, indices = index.search(query_embedding, min(k, len(self._metadata)))
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if score_threshold is not None and float(score) < score_threshold:
                continue
            item = dict(self._metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> None:
        """Persist FAISS index and metadata to disk."""
        idx_path = index_path or self.index_path
        meta_path = metadata_path or self.metadata_path
        if idx_path is None or meta_path is None:
            raise ValueError("index_path and metadata_path must be set to save")
        idx_path = Path(idx_path)
        meta_path = Path(metadata_path)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss = _import_faiss()
        faiss.write_index(self._get_index(), str(idx_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        logger.info("Saved FAISS index to %s and metadata to %s", idx_path, meta_path)

    def load(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> None:
        """Load FAISS index and metadata from disk."""
        idx_path = index_path or self.index_path
        meta_path = metadata_path or self.metadata_path
        if idx_path is None or meta_path is None:
            raise ValueError("index_path and metadata_path must be set to load")
        idx_path = Path(idx_path)
        meta_path = Path(metadata_path)
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index or metadata not found: {idx_path}, {meta_path}")
        faiss = _import_faiss()
        self._index = faiss.read_index(str(idx_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        logger.info("Loaded FAISS index from %s (%d vectors)", idx_path, len(self._metadata))

    @property
    def size(self) -> int:
        return len(self._metadata) if self._metadata else (self._index.ntotal if self._index else 0)
