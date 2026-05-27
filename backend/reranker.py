"""Cross-encoder reranker for two-stage retrieval.

Bi-encoders are fast but less accurate; cross-encoders see query+doc together
and score relevance more precisely.  Standard pattern: retrieve N>K candidates
via bi-encoder, then rerank with cross-encoder and keep top K.
"""
from __future__ import annotations

from typing import Any
import logging

log = logging.getLogger("vectorvault.reranker")


class Reranker:
    """Lazy-loaded cross-encoder for reranking search candidates."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            log.info("Loading cross-encoder: %s", self.model_name)
            from sentence_transformers import CrossEncoder  # lazy import
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = 5
    ) -> list[dict]:
        """Re-score candidates by query relevance and return top_k sorted by score."""
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
