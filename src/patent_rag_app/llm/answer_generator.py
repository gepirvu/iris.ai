"""LLM-powered answer generation from retrieved patent chunks."""

from __future__ import annotations

import re
from typing import Any

import ollama
from ollama import ResponseError

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings
from patent_rag_app.retrieval.service import ChunkSearchResult

LOGGER = get_logger(__name__)

ANSWER_GENERATION_PROMPT = """You are a patent research assistant. Based on the following patent chunks, provide a comprehensive answer to the user's question.

Question: {query}

Relevant Patent Content:
{context}

Instructions:
1. Provide a clear, comprehensive answer based on the patent content
2. Include specific references using the format [PatentID:Section] (e.g., [EP1577413_A1:Claims])
3. If information spans multiple patents, mention all relevant sources
4. Be specific about technical details, measurements, and compositions when available
5. If the context doesn't fully answer the question, acknowledge limitations

Answer:"""


class AnswerGenerator:
    """Generates LLM answers from retrieved patent chunks using Ollama."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        LOGGER.info(
            "Initialized answer generator",
            extra={
                "provider": self.settings.llm_provider,
                "model": self.settings.ollama_model,
                "url": self.settings.ollama_url,
            },
        )

    def generate_answer(self, query: str, chunks: list[ChunkSearchResult]) -> dict[str, Any]:
        """Generate an LLM answer from retrieved chunks with citations."""
        if not chunks:
            return {
                "answer": "No relevant patent content was found for your question.",
                "citations": [],
                "confidence": 0.0,
            }

        # Build context from chunks
        context_parts = []
        citations = []

        for i, chunk in enumerate(chunks, 1):
            citation = f"{chunk.patent_id}:{chunk.section_label}"
            citations.append(citation)
            context_parts.append(f"[{citation}] {chunk.text}")

        context = "\n\n".join(context_parts)

        # Generate prompt
        prompt = ANSWER_GENERATION_PROMPT.format(
            query=query,
            context=context,
        )

        try:
            # Generate answer using Ollama
            response = ollama.generate(
                model=self.settings.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for factual answers
                    "top_p": 0.9,
                    "num_predict": 500,  # Limit response length
                },
            )

            answer = response["response"].strip()

            # Extract citations from the generated answer
            used_citations = self._extract_citations_from_answer(answer, citations)

            # Calculate confidence based on chunk scores
            confidence = sum(chunk.score for chunk in chunks[:3]) / min(3, len(chunks))

            LOGGER.info(
                "Generated answer",
                extra={
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "chunks_used": len(chunks),
                    "citations_found": len(used_citations),
                    "confidence": confidence,
                },
            )

            return {
                "answer": answer,
                "citations": used_citations,
                "confidence": confidence,
            }

        except ResponseError as e:
            LOGGER.error("Ollama response error", extra={"error": str(e)})
            return {
                "answer": "Sorry, I encountered an error generating an answer. Please try again.",
                "citations": citations[:3],  # Return first 3 as fallback
                "confidence": 0.0,
            }
        except Exception as e:
            LOGGER.error("Answer generation error", extra={"error": str(e)})
            return {
                "answer": "An unexpected error occurred while generating the answer.",
                "citations": citations[:3],
                "confidence": 0.0,
            }

    def _extract_citations_from_answer(self, answer: str, available_citations: list[str]) -> list[str]:
        """Extract citation references from the generated answer."""
        # Find patterns like [EP1577413_A1:Claims] in the answer
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, answer)

        # Filter to only include valid citations that exist in our context
        used_citations = []
        for citation in found_citations:
            if citation in available_citations and citation not in used_citations:
                used_citations.append(citation)

        # If no citations were found in the answer, return the top 3 source citations
        if not used_citations:
            used_citations = available_citations[:3]

        return used_citations