"""
RAG (Retrieval-Augmented Generation) service for legal drafts.
Uses FAISS index or Postgres pgvector (legal_ai.embeddings) and embedding model to retrieve similar past drafts.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import get_settings
from app.utils.prompt_builder import build_search_query
from app.vectorstore.faiss_index import FAISSLegalIndex
from app.vectorstore.pgvector_index import PgVectorLegalIndex

logger = logging.getLogger(__name__)


def get_embedding_model(use_instructor: bool = False):
    """Load embedding model (BGE-small or Instructor)."""
    if use_instructor:
        try:
            from instructor_embedding import INSTRUCTOR
            # Instructor-xl model name
            return INSTRUCTOR("hkunlp/instructor-xl")
        except ImportError:
            logger.warning("instructor-embedding not installed; falling back to sentence-transformers")
    from sentence_transformers import SentenceTransformer
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(model, texts: list[str], instruction: Optional[str] = None) -> np.ndarray:
    """Get embeddings for a list of texts. For Instructor, optional instruction for query."""
    if instruction and hasattr(model, "encode"):
        try:
            return model.encode([[instruction, t] for t in texts], normalize_embeddings=True)
        except Exception:
            pass
    out = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return out if isinstance(out, np.ndarray) else np.array(out)


class RAGService:
    """
    Retrieves similar legal drafts from FAISS or Postgres pgvector for RAG.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        embedding_dim: Optional[int] = None,
        use_instructor: bool = False,
        use_pgvector: Optional[bool] = None,
        database_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.index_path = index_path or settings.vector_index_path
        self.metadata_path = metadata_path or settings.vector_metadata_path
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self.use_instructor = use_instructor or settings.use_instructor
        self.use_pgvector = use_pgvector if use_pgvector is not None else getattr(settings, "use_pgvector", False)
        self.database_url = database_url or getattr(settings, "database_url", None)
        self._index: Optional[Any] = None
        self._embedding_model = None

    def _get_index(self) -> Any:
        if self._index is None:
            if self.use_pgvector and self.database_url:
                self._index = PgVectorLegalIndex(dimension=self.embedding_dim, database_url=self.database_url)
                logger.info("Using pgvector (legal_ai.embeddings) for RAG")
            else:
                self._index = FAISSLegalIndex(dimension=self.embedding_dim)
                idx_path = Path(self.index_path)
                meta_path = Path(self.metadata_path)
                if idx_path.exists() and meta_path.exists():
                    self._index.load(idx_path, meta_path)
                else:
                    logger.warning("FAISS index not found; RAG will return no results until index is built")
        return self._index

    def _get_embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model(self.use_instructor)
        return self._embedding_model

    def retrieve(
        self,
        document_type: str,
        case_facts: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        **kwargs: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve similar drafts for the given document_type and case_facts.
        kwargs can include court_name, section, etc. for richer query.
        """
        settings = get_settings()
        k = top_k or settings.rag_top_k
        threshold = score_threshold if score_threshold is not None else settings.rag_score_threshold

        query_str = build_search_query(document_type, case_facts, **kwargs)
        model = self._get_embedding_model()
        query_embedding = embed_texts(
            model,
            [query_str],
            instruction="Represent the legal document query for retrieval: " if self.use_instructor else None,
        )
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        index = self._get_index()
        if index.size == 0:
            return []
        return index.search(query_embedding, k=k, score_threshold=threshold)
