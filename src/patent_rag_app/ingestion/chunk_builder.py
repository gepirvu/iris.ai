"""Patent chunk builder."""

from __future__ import annotations

import uuid
from typing import List

from patent_rag_app.ingestion.schemas import (
    ClaimRecord,
    ChunkDocument,
    PatentFrontMatter,
    ParagraphRecord,
    TableRecord,
)

FIELD_LABEL = {
    "11": "Publication number",
    "12": "Document kind",
    "19": "Office",
    "21": "Application number",
    "22": "Filing date",
    "30": "Priority",
    "43": "Publication date",
    "45": "Grant publication",
    "51": "Classification",
    "54": "Title",
    "56": "Citation",
    "57": "Abstract",
    "71": "Applicant",
    "72": "Inventors",
    "73": "Assignee",
    "74": "Representative",
    "84": "Designated states",
    "86": "International application",
    "87": "International publication",
}

FRONT_FIELD_ORDER = list(FIELD_LABEL)
MAX_PARAGRAPH_WORDS = 420
PARAGRAPH_OVERLAP_WORDS = 60


def _normalize(text: str) -> str:
    normalized = " ".join(text.split())
    return normalized[:8000]


def build_patent_chunks(document: PatentFrontMatter) -> List[ChunkDocument]:
    chunks: List[ChunkDocument] = []
    for code in FRONT_FIELD_ORDER:
        value = document.front_page_fields.get(code)
        if not value:
            continue

        chunk_key = f"front_{document.patent_id}_{code}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))

        chunk = ChunkDocument(
            chunk_id=chunk_id,
            chunk_key=chunk_key,
            patent_id=document.patent_id,
            section_code=code,
            section_label=FIELD_LABEL.get(code, f"Front matter ({code})"),
            text=_normalize(value),
            content_type="front_matter",
            paragraph_ids=[code],
            page_start=1,
            page_end=1,
        )
        chunks.append(chunk)

    paragraph_chunks = _build_description_paragraph_chunks(document.patent_id, document.description_paragraphs)
    chunks.extend(paragraph_chunks)

    claim_chunks = _build_claim_chunks(document.patent_id, document.claims)
    chunks.extend(claim_chunks)

    table_chunks = _build_table_chunks(document.patent_id, document.tables)
    chunks.extend(table_chunks)

    return chunks


def _build_claim_chunks(patent_id: str, claims: List[ClaimRecord]) -> List[ChunkDocument]:
    chunks: List[ChunkDocument] = []
    for claim in claims:
        chunk_key = f"claim_{patent_id}_{claim.claim_id}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))
        label = f"Claim {claim.number}" if claim.number else "Claim"

        chunk = ChunkDocument(
            chunk_id=chunk_id,
            chunk_key=chunk_key,
            patent_id=patent_id,
            section_code=claim.claim_id,
            section_label=label,
            text=_normalize(claim.text),
            content_type="claim",
            paragraph_ids=[claim.claim_id],
            page_start=claim.page_start,
            page_end=claim.page_end,
        )
        chunks.append(chunk)

    return chunks


def _split_with_overlap(text: str, *, max_words: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


def _build_description_paragraph_chunks(patent_id: str, paragraphs: List[ParagraphRecord]) -> List[ChunkDocument]:
    chunks: List[ChunkDocument] = []
    for paragraph in paragraphs:
        label = paragraph.label or f"Paragraph {paragraph.section_id.split('_')[-1]}"
        parts = _split_with_overlap(
            paragraph.text,
            max_words=MAX_PARAGRAPH_WORDS,
            overlap=PARAGRAPH_OVERLAP_WORDS,
        )
        if not parts:
            continue

        for index, part_text in enumerate(parts, start=1):
            suffix = "" if len(parts) == 1 else f"_part{index}"
            chunk_key = f"desc_{patent_id}_{paragraph.section_id}{suffix}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))
            part_label = label if len(parts) == 1 else f"{label} (part {index})"

            chunk = ChunkDocument(
                chunk_id=chunk_id,
                chunk_key=chunk_key,
                patent_id=patent_id,
                section_code=paragraph.section_id,
                section_label=part_label,
                text=_normalize(part_text),
                content_type="description_paragraph",
                paragraph_ids=[paragraph.section_id],
                page_start=paragraph.page_start,
                page_end=paragraph.page_end,
            )
            chunks.append(chunk)

    return chunks


def _build_table_chunks(patent_id: str, tables: List[TableRecord]) -> List[ChunkDocument]:
    chunks: List[ChunkDocument] = []
    for table in tables:
        chunk_key = f"table_{patent_id}_{table.table_id}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))
        label = table.caption or f"Table {table.table_id.split('_')[-1]}"
        text = _normalize(_summarize_table(table))
        chunk = ChunkDocument(
            chunk_id=chunk_id,
            chunk_key=chunk_key,
            patent_id=patent_id,
            section_code=table.table_id,
            section_label=label,
            text=text,
            content_type="table",
            paragraph_ids=[table.table_id],
            page_start=table.page_number,
            page_end=table.page_number,
        )
        chunks.append(chunk)
    return chunks


def _summarize_table(table: TableRecord) -> str:
    lines: List[str] = []
    headers = [heading.strip() for heading in (table.headers or [])]
    headers = [heading if heading else f"Column {idx + 1}" for idx, heading in enumerate(headers)]

    if table.caption:
        lines.append(table.caption)
    if headers:
        lines.append(" | ".join(headers))

    for row in table.rows:
        row_pairs = []
        for idx, cell in enumerate(row):
            if not cell:
                continue
            column = headers[idx] if idx < len(headers) else f"Column {idx + 1}"
            row_pairs.append(f"{column}: {cell}")
        if row_pairs:
            lines.append(", ".join(row_pairs))

    if not lines and table.rows:
        for row in table.rows:
            cells = [cell for cell in row if cell]
            if cells:
                lines.append(" | ".join(cells))

    return " ; ".join(lines)
