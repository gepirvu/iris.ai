"""Cross-encoder reranking utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

from sentence_transformers import CrossEncoder

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings

LOGGER = get_logger(__name__)


class CrossEncoderReranker:
    """Wrap a sentence-transformers cross-encoder for relevance scoring."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.reranker_model:
            raise ValueError("reranker_model setting must be configured to enable reranking")
        self.model = _load_cross_encoder(self.settings.reranker_model)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        LOGGER.info(
            "Reranking candidates",
            extra={"count": len(pairs), "model": self.settings.reranker_model},
        )
        scores = self.model.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    LOGGER.info("Loading cross-encoder", extra={"model": model_name})
    return CrossEncoder(model_name)

