"""Ingestion pipeline for patent metadata."""

from __future__ import annotations

from pathlib import Path

from patent_rag_app.config.logging import get_logger
from patent_rag_app.ingestion.parsers.pdf_parser import parse_patent_pdf
from patent_rag_app.db.patent_repository import PatentRepository
from patent_rag_app.llm.available_patents_summarizer import PatentSummarizer

LOGGER = get_logger(__name__)


class PatentIngestor:
    """Ingest full patent metadata for downstream pipelines."""

    def __init__(self, repository: PatentRepository | None = None) -> None:
        self.repository = repository or PatentRepository()
        self.summarizer = PatentSummarizer()

    def ingest_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)

        LOGGER.info("Ingesting patent document", extra={"path": str(path)})

        document = parse_patent_pdf(path)
        description_payload = [
            {
                "section_id": section.section_id,
                "label": section.title,
                "text": section.text,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "section_path": section.section_path,
            }
            for section in document.sections
        ]
        claims_payload = [
            {
                "claim_id": claim.claim_id,
                "number": claim.number,
                "text": claim.text,
                "page_start": claim.page_start,
                "page_end": claim.page_end,
                "language": claim.language,
            }
            for claim in document.claims
        ]
        tables_payload = [
            {
                "table_id": table.table_id,
                "caption": table.caption,
                "page_number": table.page_number,
                "headers": table.headers,
                "rows": table.rows,
            }
            for table in document.tables
        ]

        # Generate patent summary using LLM
        title = document.front_page_fields.get("54")  # Title field
        abstract = document.front_page_fields.get("57")  # Abstract field

        # Get first paragraph for context (usually [0001])
        first_paragraph = None
        if document.sections:
            # Find first paragraph that contains meaningful content
            for section in document.sections:
                if section.text and len(section.text.strip()) > 50:
                    first_paragraph = section.text
                    break

        summary = self.summarizer.summarize_patent(
            title=title,
            abstract=abstract,
            first_paragraph=first_paragraph,
        )

        self.repository.upsert_patent(
            {
                "patent_id": document.patent_id,
                "total_pages": document.total_pages,
                "front_page_fields": document.front_page_fields,
                "source_path": document.source_path,
                "raw_text": document.raw_text,
                "summary": summary,
                "description_paragraphs": description_payload,
                "claims": claims_payload,
                "tables": tables_payload,
            }
        )

    def ingest_directory(self, directory: Path, limit: int | None = None) -> None:
        pdf_files = sorted(directory.glob("*.pdf"))
        if limit is not None:
            pdf_files = pdf_files[:limit]

        for pdf_file in pdf_files:
            self.ingest_file(pdf_file)
