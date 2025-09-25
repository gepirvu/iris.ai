"""Schemas for ingestion pipeline."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ParagraphRecord(BaseModel):
    section_id: str
    label: str
    text: str
    page_start: int
    page_end: int
    section_path: List[str] = Field(default_factory=list)



class ClaimRecord(BaseModel):
    claim_id: str
    number: str
    text: str
    page_start: int
    page_end: int
    language: str = "english"


class TableRecord(BaseModel):
    table_id: str
    caption: str | None = None
    page_number: int
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]]


class PatentFrontMatter(BaseModel):
    patent_id: str
    total_pages: int
    front_page_fields: dict[str, str]
    source_path: str
    raw_text: str | None = None
    summary: str | None = None
    description_paragraphs: List[ParagraphRecord] = Field(default_factory=list)
    claims: List[ClaimRecord] = Field(default_factory=list)
    tables: List[TableRecord] = Field(default_factory=list)


class ChunkDocument(BaseModel):
    chunk_id: str
    chunk_key: str
    patent_id: str
    section_code: str
    section_label: str
    text: str
    content_type: str
    paragraph_ids: list[str]
    page_start: int
    page_end: int



