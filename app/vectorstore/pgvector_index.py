"""
PostgreSQL pgvector-based vector store for legal document retrieval.
Uses shared DB (legal_ai.embeddings with embedding vector(384)) for RAG.
"""

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_connection(database_url: str):
    import psycopg
    from pgvector.psycopg import register_vector
    conn = psycopg.connect(database_url)
    register_vector(conn)
    return conn


class PgVectorLegalIndex:
    """
    pgvector index for legal draft embeddings in legal_ai.embeddings.
    Same interface as FAISSLegalIndex for drop-in use in RAGService.
    """

    def __init__(
        self,
        dimension: int,
        database_url: str,
    ):
        self.dimension = dimension
        self.database_url = database_url

    def _vector_str(self, vec: np.ndarray) -> str:
        if vec.dtype != np.float64:
            vec = vec.astype(np.float64)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        return "[" + ",".join(str(float(x)) for x in vec.ravel()) + "]"

    @property
    def size(self) -> int:
        with _get_connection(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT COUNT(*) FROM legal_ai.embeddings WHERE embedding IS NOT NULL'
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search legal_ai.embeddings by cosine similarity (pgvector <=>).
        Returns list of dicts with draft, document_type, facts, score.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        vec_str = self._vector_str(query_embedding.astype(np.float64))

        with _get_connection(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH q AS (SELECT %s::vector AS v)
                    SELECT
                        e.content_type,
                        e.content_id,
                        1 - (e.embedding <=> q.v) AS score,
                        d.draft_text,
                        d.facts,
                        d.document_type
                    FROM legal_ai.embeddings e
                    LEFT JOIN legal_ai.legal_drafts d
                        ON e.content_type = 'legal_draft' AND e.content_id = d.id
                    CROSS JOIN q
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <=> q.v
                    LIMIT %s
                    """,
                    (vec_str, k),
                )
                rows = cur.fetchall()

        results = []
        for row in rows:
            content_type, content_id, score, draft_text, facts, document_type = row
            if score_threshold is not None and (score or 0) < score_threshold:
                continue
            results.append({
                "content_type": content_type,
                "content_id": content_id,
                "draft": draft_text or "",
                "document_type": document_type or "",
                "facts": facts or "",
                "score": float(score) if score is not None else 0.0,
            })
        return results

    def load(self, index_path: Optional[Any] = None, metadata_path: Optional[Any] = None) -> None:
        """No-op: data lives in Postgres."""
        pass
