"""Integration test verifying Qdrant chunk indexing."""

from __future__ import annotations

import pytest

from patent_rag_app.config.settings import get_settings
from patent_rag_app.db.chunk_repository import ChunkRepository
from patent_rag_app.retrieval.qdrant_store import QdrantStore

pytestmark = pytest.mark.integration


def test_qdrant_chunks_indexed() -> None:
    settings = get_settings()
    repo = ChunkRepository(settings=settings)
    chunks = repo.fetch_all()
    assert chunks, "Expected chunks in Mongo; run chunk builder first"
    print(f"Mongo chunks: {len(chunks)}")

    store = QdrantStore(settings=settings)
    count = store.client.count(collection_name=store.collection, exact=True).count
    print(f"Qdrant points: {count}")
    assert count >= len(chunks), "Qdrant point count should cover chunk count"

    sample_chunk = chunks[0]
    points = store.client.retrieve(
        collection_name=store.collection,
        ids=[sample_chunk.chunk_id],
        with_vectors=True,
    )
    assert points, "Expected to retrieve stored chunk point"
    payload = points[0].payload
    assert payload.get("chunk_key") == sample_chunk.chunk_key
    assert payload.get("patent_id") == sample_chunk.patent_id

    vector = points[0].vector
    assert vector, "Vector should be present in retrieval response"
    print("Sample vector dims (first 5):", vector[:5])
