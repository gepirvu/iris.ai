"""Index patent chunks into Qdrant."""

from __future__ import annotations

import argparse

from patent_rag_app.config.logging import configure_logging, get_logger
from patent_rag_app.config.settings import get_settings
from patent_rag_app.db.chunk_repository import ChunkRepository
from patent_rag_app.llm.embeddings import EmbeddingEncoder
from patent_rag_app.retrieval.qdrant_store import QdrantStore

configure_logging()
LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index patent chunks into Qdrant")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks processed")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embedding batch size",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection before indexing",
    )
    args = parser.parse_args()

    # Initialize components
    settings = get_settings()
    chunk_repo = ChunkRepository()
    encoder = EmbeddingEncoder()
    qdrant_store = QdrantStore()

    batch_size = args.batch_size or settings.embedding_batch_size

    if args.recreate:
        LOGGER.info("Recreating Qdrant collection")
        # Get vector size from encoder
        vector_size = encoder.vector_size()
        qdrant_store.recreate_collection(vector_size)

    # Fetch chunks to index
    chunks = list(chunk_repo.fetch_all())
    if args.limit:
        chunks = chunks[:args.limit]

    if not chunks:
        print("No chunks found to index.")
        return

    total_chunks = len(chunks)
    indexed = 0
    warnings = []

    # Process chunks in batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]

        try:
            # Generate embeddings
            texts = [chunk.text for chunk in batch]
            embeddings = encoder.encode(texts)

            # Index to Qdrant
            qdrant_store.upsert(batch, embeddings)

            indexed += len(batch)
            LOGGER.info("Indexed batch", extra={"batch_size": len(batch), "total_indexed": indexed})

        except Exception as e:
            error_msg = f"Failed to index batch {i//batch_size + 1}: {e}"
            warnings.append(error_msg)
            LOGGER.error(error_msg)

    print(f"Indexed {indexed} / {total_chunks} chunks into Qdrant.")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
