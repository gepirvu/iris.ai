"""Qdrant storage helpers."""

from __future__ import annotations

from typing import Sequence

from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings
from patent_rag_app.db.qdrant_client import get_qdrant_client
from patent_rag_app.ingestion.schemas import ChunkDocument

LOGGER = get_logger(__name__)


class QdrantStore:
    """Wrapper around Qdrant client for chunk indexing and search."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = get_qdrant_client(self.settings)
        self.collection = self.settings.collection_name

    def recreate_collection(self, vector_size: int) -> None:
        LOGGER.info(
            "Recreating Qdrant collection",
            extra={"collection": self.collection, "vector_size": vector_size},
        )
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    def upsert(self, chunks: Sequence[ChunkDocument], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        self._ensure_payload_indexes()
        points = [
            qmodels.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        LOGGER.info(
            "Upserting points in Qdrant",
            extra={"collection": self.collection, "points": len(points)},
        )
        self.client.upsert(collection_name=self.collection, points=points)

    def _ensure_payload_indexes(self) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="patent_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:  # index already exists or collection missing
            if getattr(exc, "status_code", None) == 409:
                return
            LOGGER.debug(
                "Unable to create payload index",
                extra={"collection": self.collection, "error": str(exc)},
            )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        query_filter: qmodels.Filter | None = None,
    ) -> list[qmodels.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_vectors=False,
            with_payload=True,
        )
