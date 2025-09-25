"""Search API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from patent_rag_app.llm.answer_generator import AnswerGenerator
from patent_rag_app.retrieval.service import PatentRetrievalService

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, description="Search term")
    patent_id: str | None = Field(default=None, description="Restrict search to this patent")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    generate_answer: bool = Field(default=False, description="Generate LLM answer from retrieved chunks")


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
    llm_answer: str | None = None
    citations: list[str] | None = None
    confidence: float | None = None


def get_retrieval_service() -> PatentRetrievalService:
    return PatentRetrievalService()


def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator()


@router.post("/chunks", response_model=SearchResponse)
def search_chunks(
    payload: SearchRequest,
    service: PatentRetrievalService = Depends(get_retrieval_service),
    answer_generator: AnswerGenerator = Depends(get_answer_generator),
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

    # Generate LLM answer if requested
    llm_answer = None
    citations = None
    confidence = None

    if payload.generate_answer and results:
        answer_data = answer_generator.generate_answer(payload.query, results)
        llm_answer = answer_data["answer"]
        citations = answer_data["citations"]
        confidence = answer_data["confidence"]

    return SearchResponse(
        query=payload.query,
        results=items,
        llm_answer=llm_answer,
        citations=citations,
        confidence=confidence,
    )

