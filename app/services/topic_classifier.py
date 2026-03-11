"""
Lightweight legal topic classifier used to enrich retrieval and routing.
"""

from enum import Enum
from typing import Optional

from app.config import get_settings
from app.models.legal_llm import LegalLLMClient


class LegalTopic(str, Enum):
    criminal = "criminal"
    civil = "civil"
    consumer = "consumer"
    labour = "labour"
    corporate = "corporate"
    property = "property"
    family = "family"
    unknown = "unknown"


class TopicClassifier:
    """
    Simple rule-based + optional LLM-based topic classifier.
    """

    def __init__(self, use_llm_fallback: bool = False) -> None:
        self.use_llm_fallback = use_llm_fallback
        self._llm: Optional[LegalLLMClient] = None

    def _get_llm(self) -> LegalLLMClient:
        if self._llm is None:
            settings = get_settings()
            self._llm = LegalLLMClient(
                base_url=settings.vllm_base_url,
                model_name=settings.llm_model_name,
                max_tokens=64,
                temperature=0.0,
                timeout=getattr(settings, "llm_timeout", 60.0),
            )
        return self._llm

    def classify(self, text: str) -> LegalTopic:
        t = text.lower()
        if any(k in t for k in ["ipc", "fir", "bail", "offence", "crime", "charge sheet", "police"]):
            return LegalTopic.criminal
        if any(k in t for k in ["divorce", "maintenance", "custody", "marriage", "domestic violence"]):
            return LegalTopic.family
        if any(k in t for k in ["consumer forum", "consumer court", "deficiency in service", "consumer protection"]):
            return LegalTopic.consumer
        if any(k in t for k in ["labour", "workman", "industrial dispute", "wages", "gratuity"]):
            return LegalTopic.labour
        if any(k in t for k in ["company", "shareholder", "director", "mca", "companies act"]):
            return LegalTopic.corporate
        if any(k in t for k in ["property", "title", "possession", "sale deed", "lease deed", "land"]):
            return LegalTopic.property
        if any(k in t for k in ["injunction", "damages", "specific performance", "civil suit"]):
            return LegalTopic.civil

        if not self.use_llm_fallback:
            return LegalTopic.unknown

        prompt = (
            "You are a classifier for Indian legal matters.\n"
            "Classify the following text into exactly one topic:\n"
            "- criminal\n- civil\n- consumer\n- labour\n- corporate\n- property\n- family\n\n"
            "Text:\n"
            f"{text}\n\n"
            "Respond with only the topic name."
        )
        llm = self._get_llm()
        raw = llm.complete(prompt)
        cleaned = raw.strip().lower()
        for topic in LegalTopic:
            if topic.value == cleaned:
                return topic
        return LegalTopic.unknown

