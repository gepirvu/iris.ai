"""Build patent content chunks and store them in MongoDB."""

from __future__ import annotations

import argparse

from patent_rag_app.config.logging import configure_logging, get_logger
from patent_rag_app.db.patent_repository import PatentRepository
from patent_rag_app.db.chunk_repository import ChunkRepository
from patent_rag_app.ingestion.chunk_builder import build_patent_chunks

configure_logging()
LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patent chunks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patents processed")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing chunks to MongoDB")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for confirmation before processing",
    )
    args = parser.parse_args()

    if not args.yes:
        message = f"About to build chunks for {args.limit or 'all available'} patents. Proceed? [y/N] "
        choice = input(message).strip().lower()
        if choice not in {"y", "yes"}:
            print("Aborted.")
            return

    # Process patents and build chunks
    patent_repo = PatentRepository()
    chunk_repo = ChunkRepository()

    patents = patent_repo.fetch_all()
    if args.limit:
        patents = patents[:args.limit]

    processed = 0
    persisted = 0
    total_chunks = 0
    warnings = []

    for patent in patents:
        try:
            chunks = build_patent_chunks(patent)
            total_chunks += len(chunks)

            if not args.dry_run:
                chunk_repo.upsert_many(chunks)
                persisted += 1

            processed += 1
            LOGGER.info("Built chunks for patent", extra={"patent_id": patent.patent_id, "chunks": len(chunks)})

        except Exception as e:
            error_msg = f"Failed to build chunks for {patent.patent_id}: {e}"
            warnings.append(error_msg)
            LOGGER.error(error_msg)

    print(f"Built chunks for {processed} patents (persisted: {persisted if not args.dry_run else 0}).")
    print(f"Chunks generated: {total_chunks}")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
