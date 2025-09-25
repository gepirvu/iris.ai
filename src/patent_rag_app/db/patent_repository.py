"""MongoDB schema and repositories."""

from __future__ import annotations

from typing import Any, Mapping

from pymongo.collection import Collection

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings
from patent_rag_app.db.mongo_client import get_database
from patent_rag_app.ingestion.schemas import PatentFrontMatter

LOGGER = get_logger(__name__)


class PatentRepository:
    """Repository for patent metadata storage."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        db = get_database(self.settings.mongo_database, settings=self.settings)
        self.collection: Collection[Any] = db[self.settings.patents_collection]
        self.collection.create_index("patent_id", unique=True)

    def upsert_patent(self, patent_document: Mapping[str, Any]) -> None:
        patent_id = patent_document["patent_id"]
        LOGGER.info("Upserting patent record", extra={"patent_id": patent_id})

        self.collection.update_one(
            {"patent_id": patent_id},
            {
                "$set": {
                    "patent_id": patent_id,
                    "total_pages": patent_document.get("total_pages"),
                    "front_page_fields": patent_document.get("front_page_fields", {}),
                    "source_path": patent_document.get("source_path"),
                    "raw_text": patent_document.get("raw_text"),
                    "summary": patent_document.get("summary"),
                    "description_paragraphs": patent_document.get("description_paragraphs", []),
                    "claims": patent_document.get("claims", []),
                    "tables": patent_document.get("tables", []),
                }
            },
            upsert=True,
        )

    def fetch_all(self) -> list[PatentFrontMatter]:
        documents = list(self.collection.find({}, {"_id": 0}))
        return [PatentFrontMatter(**doc) for doc in documents]

    def fetch_by_id(self, patent_id: str) -> PatentFrontMatter | None:
        document = self.collection.find_one({"patent_id": patent_id}, {"_id": 0})
        if document is None:
            return None
        return PatentFrontMatter(**document)
