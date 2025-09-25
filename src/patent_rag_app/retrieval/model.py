"""Chunk models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    patent_id: str
    section_code: str
    section_label: str
    text: str
    content_type: str
    paragraph_ids: list[str]
    page_start: int
    page_end: int

    def as_payload(self) -> dict[str, str | int | list[str]]:
        return {
            "chunk_id": self.chunk_id,
            "patent_id": self.patent_id,
            "section_code": self.section_code,
            "section_label": self.section_label,
            "content_type": self.content_type,
            "paragraph_ids": self.paragraph_ids,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
