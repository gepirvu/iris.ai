"""Embedding utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings

LOGGER = get_logger(__name__)


class EmbeddingEncoder:
    """Wraps sentence-transformers models for chunk embedding."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = SentenceTransformer(self.settings.embedding_model)
        LOGGER.info(
            "Loaded embedding model",
            extra={"model": self.settings.embedding_model},
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        LOGGER.info("Encoding batch", extra={"size": len(texts)})
        return self.model.encode(texts, batch_size=self.settings.embedding_batch_size, convert_to_numpy=True)

    def encode_iter(self, texts: Iterable[str], batch_size: int | None = None) -> Iterable[np.ndarray]:
        batch: list[str] = []
        target_size = batch_size or self.settings.embedding_batch_size

        for text in texts:
            batch.append(text)
            if len(batch) >= target_size:
                yield self.encode(batch)
                batch.clear()

        if batch:
            yield self.encode(batch)

    def vector_size(self) -> int:
        """Get the dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()
