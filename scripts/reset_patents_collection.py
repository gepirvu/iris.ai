"""Utility to clear the patents collection in MongoDB."""

from __future__ import annotations

import argparse

from patent_rag_app.config.logging import configure_logging, get_logger
from patent_rag_app.config.settings import get_settings
from patent_rag_app.db.mongo_client import get_database

configure_logging()
LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the patents collection")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    settings = get_settings()
    db = get_database(settings.mongo_database, settings=settings)
    collection = db[settings.patents_collection]

    doc_count = collection.count_documents({})
    if doc_count == 0:
        print("Collection is already empty.")
        return

    if not args.force:
        message = (
            f"About to delete {doc_count} documents from "
            f"'{settings.mongo_database}.{settings.patents_collection}'. Proceed? [y/N] "
        )
        if input(message).strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return

    LOGGER.warning(
        "Clearing patents collection",
        extra={
            "database": settings.mongo_database,
            "collection": settings.patents_collection,
            "documents": doc_count,
        },
    )
    result = collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents.")


if __name__ == "__main__":
    main()
