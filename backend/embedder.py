"""
embedder.py — Sentence-transformer embedding wrapper for VectorVault.

Uses the lightweight all-MiniLM-L6-v2 model (22 M parameters, 384-dim
embeddings) which runs comfortably on CPU.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper around a SentenceTransformer model.

    The model is loaded once at construction time and reused for all calls.
    All returned embeddings are L2-normalised so that inner-product search is
    equivalent to cosine similarity.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Parameters
        ----------
        texts:
            List of strings to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), embedding_dim)``, dtype ``float32``,
            L2-normalised.
        """
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalisation
        ).astype(np.float32)

        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns
        -------
        np.ndarray
            Shape ``(1, embedding_dim)``, dtype ``float32``, L2-normalised.
        """
        return self.embed([query])

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (384 for all-MiniLM-L6-v2)."""
        return self._model.get_sentence_embedding_dimension()
