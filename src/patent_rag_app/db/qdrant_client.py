"""Qdrant client utilities."""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def _cached_qdrant_client(url: str, api_key: str | None) -> QdrantClient:
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)


def get_qdrant_client(settings: AppSettings | None = None) -> QdrantClient:
    cfg = settings or get_settings()

    if cfg.use_qdrant_cloud:
        LOGGER.info("Connecting to Qdrant Cloud", extra={"url": cfg.sanitize_uri(cfg.qdrant_url)})
        if settings is None:
            return _cached_qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key)
        return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    LOGGER.info("Connecting to local Qdrant", extra={"url": cfg.qdrant_url})
    if settings is None:
        return _cached_qdrant_client(cfg.qdrant_url, None)
    return QdrantClient(url=cfg.qdrant_url)
