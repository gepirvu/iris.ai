"""CLI entrypoint for ingesting patent metadata."""

from __future__ import annotations

from pathlib import Path

import click

from patent_rag_app.config.logging import configure_logging, get_logger
from patent_rag_app.ingestion.patent_ingestor import PatentIngestor

configure_logging()
LOGGER = get_logger(__name__)


@click.command()
@click.argument("source", type=click.Path(path_type=Path))
@click.option("--limit", type=int, default=None, help="Limit number of PDFs to ingest")
@click.option("--stop-on-error", is_flag=True, help="Abort after the first ingestion failure")
def ingest_patents(source: Path, limit: int | None, stop_on_error: bool) -> None:
    """Ingest patent metadata from a PDF file or directory."""
    ingestor = PatentIngestor()

    if source.is_file():
        ingestor.ingest_file(source)
        LOGGER.info("Ingestion complete", extra={"file": str(source)})
        click.echo(f"Ingested 1 PDF from {source}.")
        return

    if not source.is_dir():
        raise click.BadParameter(f"Unsupported source path: {source}")

    # Process directory of PDFs
    pdf_files = sorted(source.glob("*.pdf"))
    if limit is not None:
        pdf_files = pdf_files[:limit]

    succeeded = 0
    warnings = []

    for pdf_path in pdf_files:
        try:
            ingestor.ingest_file(pdf_path)
            succeeded += 1
            LOGGER.info("Ingested PDF", extra={"file": str(pdf_path)})
        except Exception as e:
            error_msg = f"Failed to ingest {pdf_path}: {e}"
            warnings.append(error_msg)
            LOGGER.error(error_msg)
            if stop_on_error:
                break

    click.echo(f"Ingested {succeeded} / {len(pdf_files)} PDFs from {source}.")
    for warning in warnings:
        click.echo(f"Warning: {warning}", err=True)


if __name__ == "__main__":
    ingest_patents()
