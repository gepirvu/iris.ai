"""Integration test ensuring Mongo patent records exist."""

from __future__ import annotations

import json
from typing import Any

import pytest

from patent_rag_app.config.settings import get_settings
from patent_rag_app.db.mongo_client import get_database


@pytest.mark.integration
def test_patent_front_matter_ingested() -> None:
    settings = get_settings()
    db = get_database(settings.mongo_database, settings=settings)
    collection = db[settings.patents_collection]

    documents = list(collection.find({}, {"_id": 0}))

    print("[Mongo] Patent documents:\n" + json.dumps(documents, indent=2))

    assert documents, "Expected at least one patent document"

    for doc in documents:
        assert "patent_id" in doc
        assert "total_pages" in doc
        assert isinstance(doc.get("front_page_fields", {}), dict)
        assert doc.get("front_page_fields"), "Front page fields should not be empty"

