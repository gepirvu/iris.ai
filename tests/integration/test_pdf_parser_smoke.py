"""Smoke tests for PDF front-matter parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from patent_rag_app.config.logging import configure_logging, get_logger
from patent_rag_app.ingestion.parsers.pdf_parser import parse_patent_pdf

configure_logging()
LOGGER = get_logger(__name__)

DATA_DIR = Path("data")
SAMPLE_FILES = [
    "EP1577413_A1.pdf",
    "EP2537954_A1.pdf",
]
EXPECTED_FIELDS = {"11", "12", "21", "22", "43", "51"}


@pytest.mark.parametrize("filename", SAMPLE_FILES)
def test_parse_front_matter(filename: str) -> None:
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Missing PDF fixture: {filename}")

    document = parse_patent_pdf(path)

    LOGGER.info(
        "Parsed %s: pages=%s fields=%s",
        document.patent_id,
        document.total_pages,
        sorted(document.front_page_fields.keys()),
    )
    LOGGER.info(
        "Field samples: 43=%s | 51=%s | 86=%s",
        document.front_page_fields.get("43", ""),
        document.front_page_fields.get("51", ""),
        document.front_page_fields.get("86", ""),
    )

    assert document.patent_id == path.stem
    assert document.total_pages > 0
    assert document.front_page_text
    assert EXPECTED_FIELDS.issubset(document.front_page_fields.keys())
    assert document.claims, "Expected claims to be parsed"

    if filename == "EP1577413_A1.pdf":
        assert "21.09.2005" in document.front_page_fields.get("43", "")
        field_51 = document.front_page_fields.get("51", "").lower()
        assert "int" in field_51
        assert "c22c" in field_51
        assert "pct/" in document.front_page_fields.get("86", "").lower()

        assert document.tables, "Expected tables to be extracted"
        first_table = document.tables[0]
        assert first_table.rows, "Expected table rows to be present"
        assert any(cell for cell in first_table.rows[0]), "Expected header cells"

        first_claim = document.claims[0]
        assert first_claim.number == "1"
        assert "fe-cr-si" in first_claim.text.lower()

    if filename == "EP2537954_A1.pdf":
        assert "int" in document.front_page_fields.get("51", "").lower()
