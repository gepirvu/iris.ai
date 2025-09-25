"""Data models for ingestion pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from pydantic import BaseModel, Field


class PatentSection(BaseModel):
    section_id: str
    title: str
    text: str
    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)

    def token_length(self) -> int:
        return len(self.text.split())


class PatentClaim(BaseModel):
    claim_id: str
    number: str
    text: str
    page_start: int
    page_end: int
    language: str = "english"


class PatentTable(BaseModel):
    table_id: str
    caption: str | None = None
    page_number: int
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]]


class PatentDocument(BaseModel):
    patent_id: str
    title: str | None = None
    abstract: str | None = None
    publication_date: datetime | None = None
    total_pages: int = 0
    front_page_text: str | None = None
    front_page_fields: dict[str, str] = Field(default_factory=dict)
    sections: list[PatentSection] = Field(default_factory=list)
    tables: list[PatentTable] = Field(default_factory=list)
    claims: list[PatentClaim] = Field(default_factory=list)
    raw_text: str = ""
    source_path: str

    def iter_section_text(self) -> Iterable[str]:
        for section in self.sections:
            yield section.text


class ChunkMetadata(BaseModel):
    chunk_id: str
    patent_id: str
    section_id: str | None
    section_title: str | None
    page_start: int
    page_end: int
    ordinal: int
    text: str
    tokens: int

    @classmethod
    def from_section(
        cls,
        *,
        patent_id: str,
        section: PatentSection,
        text: str,
        ordinal: int,
        page_start: int,
        page_end: int,
        chunk_id: str,
    ) -> "ChunkMetadata":
        return cls(
            chunk_id=chunk_id,
            patent_id=patent_id,
            section_id=section.section_id,
            section_title=section.title,
            page_start=page_start,
            page_end=page_end,
            ordinal=ordinal,
            text=text,
            tokens=len(text.split()),
        )


class IngestionResult(BaseModel):
    patent_id: str
    chunk_count: int
    section_count: int
    table_count: int
    qdrant_points: int
    mongo_documents: int
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return self.model_dump()

