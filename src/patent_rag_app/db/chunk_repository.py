"""Chunk repository for MongoDB."""

from __future__ import annotations

import re
from typing import Iterable, Iterator, List, Optional, Tuple

from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings
from patent_rag_app.db.mongo_client import get_database
from patent_rag_app.ingestion.schemas import ChunkDocument

LOGGER = get_logger(__name__)


class ChunkRepository:
    """Persist and query chunk documents."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        db = get_database(self.settings.mongo_database, settings=self.settings)
        self.collection: Collection = db[self.settings.chunks_collection]
        self.collection.create_index("chunk_key", unique=True)
        self.collection.create_index([("patent_id", 1), ("section_code", 1)])
        self.collection.create_index([("content_type", 1)])
        self.collection.create_index([("text", "text")])

    def _bulk_operations(self, chunks: Iterable[ChunkDocument]) -> Iterator[UpdateOne]:
        for chunk in chunks:
            doc = chunk.model_dump()
            yield UpdateOne(
                {"chunk_key": doc["chunk_key"]},
                {"$set": doc},
                upsert=True,
            )

    def upsert_many(self, chunks: Iterable[ChunkDocument]) -> None:
        ops = list(self._bulk_operations(chunks))
        if not ops:
            return

        LOGGER.info(
            "Upserting chunks",
            extra={"count": len(ops), "collection": self.settings.chunks_collection},
        )
        try:
            self.collection.bulk_write(ops, ordered=False)
        except BulkWriteError as exc:  # pragma: no cover
            LOGGER.error("Bulk write failed", extra={"error": exc.details})
            raise

    def fetch_by_patent(self, patent_id: str) -> List[ChunkDocument]:
        cursor = self.collection.find({"patent_id": patent_id}, {"_id": 0})
        return [ChunkDocument(**doc) for doc in cursor]

    def fetch_all(self) -> List[ChunkDocument]:
        cursor = self.collection.find({}, {"_id": 0})
        return [ChunkDocument(**doc) for doc in cursor]

    def fetch_by_chunk_key(self, chunk_key: str) -> Optional[ChunkDocument]:
        document = self.collection.find_one({"chunk_key": chunk_key}, {"_id": 0})
        if document is None:
            return None
        return ChunkDocument(**document)

    def fetch_by_section(self, patent_id: str, section_code: str) -> Optional[ChunkDocument]:
        document = self.collection.find_one(
            {"patent_id": patent_id, "section_code": section_code},
            {"_id": 0},
        )
        if document is None:
            return None
        return ChunkDocument(**document)

    def fetch_claims(self, patent_id: str, *, limit: int | None = None) -> List[ChunkDocument]:
        query = {
            "patent_id": patent_id,
            "content_type": "claim",
            "section_code": {
                "$regex": re.compile(r"^claim_\d{4}$"),
                "$ne": "claim_0000",
            },
        }
        cursor = self.collection.find(query, {"_id": 0}).sort("section_code", 1)
        if limit is not None:
            cursor = cursor.limit(limit)
        return [ChunkDocument(**doc) for doc in cursor]

    def count_claims(self, patent_id: str) -> int:
        return int(self.collection.count_documents({"patent_id": patent_id, "content_type": "claim"}))

    def text_search(
        self,
        query: str,
        *,
        patent_id: str | None = None,
        limit: int = 20,
    ) -> List[tuple[float, ChunkDocument]]:
        if not query:
            return []
        filter_query: dict[str, object] = {"$text": {"$search": query}}
        if patent_id:
            filter_query["patent_id"] = patent_id
        projection = {"_id": 0, "score": {"$meta": "textScore"}}
        cursor = (
            self.collection.find(filter_query, projection)
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )
        hits: List[tuple[float, ChunkDocument]] = []
        for doc in cursor:
            score = float(doc.pop("score", 0.0))
            hits.append((score, ChunkDocument(**doc)))
        return hits

    def clear(self) -> None:
        result = self.collection.delete_many({})
        LOGGER.info(
            "Cleared chunks collection",
            extra={"collection": self.settings.chunks_collection, "deleted": result.deleted_count},
        )
