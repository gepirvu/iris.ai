"""Search API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from patent_rag_app.retrieval.service import PatentRetrievalService

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, description="Search term")
    patent_id: str | None = Field(default=None, description="Restrict search to this patent")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class SourceItem(BaseModel):
    patent_id: str
    section_label: str
    chunk_id: str
    chunk_key: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SourceItem]


def get_retrieval_service() -> PatentRetrievalService:
    return PatentRetrievalService()


@router.post("/chunks", response_model=SearchResponse)
def search_chunks(
    payload: SearchRequest,
    service: PatentRetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    results = service.search(
        payload.query,
        top_k=payload.top_k,
        patent_id=payload.patent_id,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No matching patent chunks found")

    items = [
        SourceItem(
            patent_id=result.patent_id,
            section_label=result.section_label,
            chunk_id=result.chunk_id,
            chunk_key=result.chunk_key,
            score=result.score,
            text=result.text,
        )
        for result in results
    ]

    return SearchResponse(query=payload.query, results=items)

