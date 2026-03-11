"""
Composite legal knowledge retrieval layer.

Combines:
- FAISS / pgvector semantic search
- Bare act sections (IPC/BNS, CrPC/BNSS, CPC, Evidence, Contract, NI Act, etc.)
- Judgments
- Draft templates and past drafts

Used as a reusable service before calling the LLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import DATASETS_DIR, get_settings
from app.services.rag_service import RAGService, embed_texts, get_embedding_model
from app.services.cache_service import CacheService
from app.services.legal_graph_service import LegalGraphService
from app.services.topic_classifier import TopicClassifier, LegalTopic
from app.vectorstore.faiss_index import FAISSLegalIndex

logger = logging.getLogger(__name__)


class LegalRetrievalService:
    """
    High-level retrieval service for legal context.
    """

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        acts_path: Optional[Path] = None,
        judgments_path: Optional[Path] = None,
        templates_path: Optional[Path] = None,
    ) -> None:
        settings = get_settings()
        self.rag = rag_service or RAGService()
        self.acts_path = acts_path or (DATASETS_DIR / "bare_acts.jsonl")
        self.judgments_path = judgments_path or (DATASETS_DIR / "judgments.jsonl")
        self.templates_path = templates_path or (DATASETS_DIR / "draft_templates.jsonl")
        self._embedding_model = None
        self._acts_cache: List[Dict[str, Any]] | None = None
        self._judgments_cache: List[Dict[str, Any]] | None = None
        self._templates_cache: List[Dict[str, Any]] | None = None
        self.embedding_dim = settings.embedding_dim
        self._topic_classifier = TopicClassifier()
        self._graph_service = LegalGraphService(
            acts_path=self.acts_path,
            judgments_path=self.judgments_path,
            templates_path=self.templates_path,
        )
        self._cache = CacheService.instance()

    # -------- Bare Act Knowledge --------

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _get_embedding_model(self):
        if self._embedding_model is None:
            settings = get_settings()
            self._embedding_model = get_embedding_model(settings.use_instructor)
        return self._embedding_model

    def _ensure_acts_loaded(self) -> List[Dict[str, Any]]:
        if self._acts_cache is None:
            self._acts_cache = self._load_jsonl(self.acts_path)
        return self._acts_cache

    def _ensure_judgments_loaded(self) -> List[Dict[str, Any]]:
        if self._judgments_cache is None:
            self._judgments_cache = self._load_jsonl(self.judgments_path)
        return self._judgments_cache

    def _ensure_templates_loaded(self) -> List[Dict[str, Any]]:
        if self._templates_cache is None:
            self._templates_cache = self._load_jsonl(self.templates_path)
        return self._templates_cache

    def get_relevant_sections(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve bare act sections relevant to a natural language query.
        Falls back to simple keyword filtering when embeddings are unavailable.
        """
        sections = self._ensure_acts_loaded()
        if not sections:
            return []

        # Simple heuristic for when no embedding backend is available
        try:
            model = self._get_embedding_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding model unavailable for sections search: %s", exc)
            q = query.lower()
            filtered = [
                s
                for s in sections
                if any(k in q for k in str(s.get("keywords", [])).lower().split())
                or any(k in q for k in str(s.get("title", "")).lower().split())
            ]
            return filtered[:top_k]

        texts = [
            f"{s.get('act_name', s.get('act'))} section {s.get('section_number', s.get('section'))}: "
            f"{s.get('title', '')}\n{s.get('text', '')}"
            for s in sections
        ]
        query_emb = embed_texts(model, [query])
        docs_emb = embed_texts(model, texts)

        # Use an in-memory FAISS index for sections only
        index = FAISSLegalIndex(dimension=self.embedding_dim)
        index.add(docs_emb, metadata=sections)
        results = index.search(query_emb, k=top_k)
        return results

    # -------- Judgments --------

    def get_relevant_judgments(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant judgments for the query using embeddings.
        """
        judgments = self._ensure_judgments_loaded()
        if not judgments:
            return []
        try:
            model = self._get_embedding_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding model unavailable for judgment search: %s", exc)
            q = query.lower()
            filtered = [
                j
                for j in judgments
                if any(
                    term in q
                    for term in str(j.get("headnotes", "")).lower().split()
                    + str(j.get("catchwords", "")).lower().split()
                )
            ]
            return filtered[:top_k]

        texts = [
            f"{j.get('case_name', '')} {j.get('citation', '')} {j.get('court', '')}\n"
            f"{j.get('facts', '')}\n{j.get('issues', '')}\n{j.get('ratio', '')}"
            for j in judgments
        ]
        query_emb = embed_texts(model, [query])
        docs_emb = embed_texts(model, texts)
        index = FAISSLegalIndex(dimension=self.embedding_dim)
        index.add(docs_emb, metadata=judgments)
        results = index.search(query_emb, k=top_k)
        return results

    # -------- Draft templates / examples --------

    def get_relevant_templates(
        self,
        query: str,
        document_type: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve draft templates or past drafts relevant to the query.
        """
        templates = self._ensure_templates_loaded()
        if document_type:
            templates = [t for t in templates if t.get("document_type") == document_type]
        if not templates:
            return []
        try:
            model = self._get_embedding_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding model unavailable for templates search: %s", exc)
            q = query.lower()
            filtered = [
                t
                for t in templates
                if q in str(t.get("facts", "")).lower() or q in str(t.get("draft", "")).lower()
            ]
            return filtered[:top_k]

        texts = [
            f"{t.get('document_type', '')}\n{t.get('facts', '')}\n{t.get('draft', '')}"
            for t in templates
        ]
        query_emb = embed_texts(model, [query])
        docs_emb = embed_texts(model, texts)
        index = FAISSLegalIndex(dimension=self.embedding_dim)
        index.add(docs_emb, metadata=templates)
        results = index.search(query_emb, k=top_k)
        return results

    # -------- Unified pipeline --------

    def retrieve_full_context(
        self,
        user_query: str,
        *,
        document_type: Optional[str] = None,
        extra_filters: Optional[Dict[str, str]] = None,
        top_k_sections: int = 5,
        top_k_judgments: int = 5,
        top_k_templates: int = 3,
    ) -> Dict[str, Any]:
        """
        Unified context retrieval pipeline:

        user_query
        -> relevant bare act sections
        -> relevant judgments
        -> relevant draft templates / past drafts (RAG)
        """
        extra_filters = extra_filters or {}

        # Topic classification (used mainly for enrichment/analytics for now)
        topic = self._topic_classifier.classify(user_query)

        cache_key = self._cache.make_key("retrieval_context", user_query, document_type, extra_filters, topic.value)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        sections = self.get_relevant_sections(user_query, top_k=top_k_sections)
        judgments = self.get_relevant_judgments(user_query, top_k=top_k_judgments)

        # Existing RAG on past drafts
        rag_examples: List[Dict[str, Any]] = []
        if document_type:
            rag_examples = self.rag.retrieve(
                document_type=document_type,
                case_facts=user_query,
                **extra_filters,
            )

        templates = self.get_relevant_templates(
            user_query,
            document_type=document_type,
            top_k=top_k_templates,
        )

        # Construct context text blocks for prompts
        context_parts: List[str] = []
        if sections:
            context_parts.append("### Relevant statutory provisions")
            for s in sections:
                act = s.get("act_name") or s.get("act") or ""
                num = s.get("section_number") or s.get("section") or ""
                title = s.get("title") or ""
                text = s.get("text", "")
                context_parts.append(f"- {act} Section {num}: {title}\n{text[:800]}")

        if judgments:
            context_parts.append("\n### Relevant judgments")
            for j in judgments[:top_k_judgments]:
                title = j.get("case_name") or j.get("title") or ""
                citation = j.get("citation") or ""
                ratio = j.get("ratio", "")
                context_parts.append(f"- {title} ({citation})\nRatio: {ratio[:800]}")

        if rag_examples:
            context_parts.append("\n### Similar drafts from knowledge base")
            for ex in rag_examples[:top_k_templates]:
                facts = ex.get("facts", "")
                draft = ex.get("draft", "")
                context_parts.append(f"- Facts: {facts[:400]}\nDraft excerpt:\n{draft[:800]}")

        if templates:
            context_parts.append("\n### Draft templates")
            for t in templates[:top_k_templates]:
                label = t.get("name") or t.get("document_type") or "Template"
                body = t.get("template") or t.get("draft") or ""
                context_parts.append(f"- {label}:\n{body[:800]}")

        full_context = "\n\n".join(context_parts)

        # Graph-based context
        graph_context = self._graph_service.build_context_snippets_for_query(user_query)

        result = {
            "sections": sections,
            "judgments": judgments,
            "rag_examples": rag_examples,
            "templates": templates,
            "context_text": full_context,
            "graph_context": graph_context,
            "topic": topic.value,
        }
        self._cache.set(cache_key, result)
        return result


def get_relevant_sections(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function exposing bare act retrieval for external callers.
    """
    service = LegalRetrievalService()
    return service.get_relevant_sections(query=query, top_k=top_k)

