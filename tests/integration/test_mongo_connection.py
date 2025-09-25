"""Integration test for MongoDB connectivity."""

from __future__ import annotations

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from patent_rag_app.config.settings import get_settings
from patent_rag_app.db.mongo_client import build_mongo_uri


def test_mongo_ping() -> None:
    """Ensure MongoDB is reachable using provided credentials."""
    settings = get_settings()
    uri = build_mongo_uri(settings)
    client = MongoClient(uri, tz_aware=True)
    try:
        response = client.admin.command("ping")
    except PyMongoError as exc:  # pragma: no cover - surfaced as test failure
        raise AssertionError(f"MongoDB ping failed: {exc}") from exc
    finally:
        client.close()

    assert response.get("ok", 0) == 1.0

