"""Utility to drop the chunks collection in MongoDB."""

from __future__ import annotations

import argparse

from patent_rag_app.config.logging import configure_logging, get_logger
from patent_rag_app.config.settings import get_settings
from patent_rag_app.db.mongo_client import get_database

configure_logging()
LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop the chunks collection")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    settings = get_settings()
    db = get_database(settings.mongo_database, settings=settings)
    collection_name = settings.chunks_collection

    if not args.force:
        message = (
            f"This will drop '{settings.mongo_database}.{collection_name}' completely. Proceed? [y/N] "
        )
        if input(message).strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return

    LOGGER.warning(
        "Dropping chunks collection",
        extra={"database": settings.mongo_database, "collection": collection_name},
    )
    db.drop_collection(collection_name)
    print("Chunks collection dropped.")


if __name__ == "__main__":
    main()
