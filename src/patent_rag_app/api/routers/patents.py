"""Patent metadata endpoints for frontend consumption."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from patent_rag_app.db.patent_repository import PatentRepository
from patent_rag_app.ingestion.schemas import PatentFrontMatter

router = APIRouter(prefix="/patents", tags=["patents"])


class PatentSummary(BaseModel):
    """Minimal information about a patent front page."""

    id: str
    patent_id: str
    title: str
    summary: str | None = None
    total_pages: int
    fields: dict[str, str]
    source_path: str | None = None


def get_patent_repository() -> PatentRepository:
    return PatentRepository()


def _derive_title(document: PatentFrontMatter) -> str:
    fields = document.front_page_fields
    return fields.get("54") or fields.get("57") or "Untitled patent"


@router.get("", response_model=List[PatentSummary])
async def list_patents(
    limit: int = Query(10, ge=1, le=100),
    repository: PatentRepository = Depends(get_patent_repository),
) -> List[PatentSummary]:
    """Return a limited slice of stored patent front-matter documents."""

    documents = repository.fetch_all()

    if not documents:
        return []

    summaries: List[PatentSummary] = []
    for document in documents[:limit]:
        summaries.append(
            PatentSummary(
                id=document.patent_id,
                patent_id=document.patent_id,
                title=_derive_title(document),
                summary=document.summary,
                total_pages=document.total_pages,
                fields=document.front_page_fields,
                source_path=document.source_path,
            )
        )

    return summaries

