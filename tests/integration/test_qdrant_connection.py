"""Integration test for Qdrant connectivity."""

from __future__ import annotations

from patent_rag_app.db.qdrant_client import get_qdrant_client


def test_qdrant_ping() -> None:
    """Ensure Qdrant responds when listing collections."""
    client = get_qdrant_client()

    response = client.get_collections()

    assert response is not None
    assert hasattr(response, "collections")
