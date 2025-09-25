"""MongoDB client utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pymongo import MongoClient

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings

LOGGER = get_logger(__name__)


def build_mongo_uri(settings: AppSettings) -> str:
    """Construct a MongoDB URI with credentials."""
    if settings.mongodb_uri.startswith("mongodb+srv"):
        return settings.mongodb_uri

    return (
        f"mongodb://{settings.mongodb_user}:{settings.mongodb_password}"  # noqa: S608
        f"@{settings.mongodb_uri}"
    )


@lru_cache(maxsize=1)
def _cached_client(uri: str) -> MongoClient[Any]:
    return MongoClient(uri, tz_aware=True)


def get_mongo_client(settings: AppSettings | None = None) -> MongoClient[Any]:
    """Return a MongoDB client, caching globally when no settings override is provided."""
    cfg = settings or get_settings()
    uri = build_mongo_uri(cfg)
    LOGGER.info("Connecting to MongoDB", extra={"uri": cfg.sanitize_uri(uri)})

    if settings is None:
        return _cached_client(uri)

    return MongoClient(uri, tz_aware=True)


def get_database(name: str, *, settings: AppSettings | None = None):
    """Convenience accessor for a specific MongoDB database."""
    client = get_mongo_client(settings)
    return client.get_database(name)
