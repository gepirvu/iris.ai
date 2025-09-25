"""Patent summarization using Ollama/Llama 3.1."""

from __future__ import annotations

import ollama
from openai import OpenAI

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings

LOGGER = get_logger(__name__)

SUMMARIZATION_PROMPT = """Patent Title: {title}
Patent Abstract: {abstract}
Background/Description: {background}

Write exactly 2 sentences explaining what this patent covers. Do not include any introduction, headers, or explanatory text. Start directly with the technical summary. Focus on the technology/invention and the problem it solves."""


class PatentSummarizer:
    """Generate concise summaries of patent documents using configurable LLM."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def summarize_patent(
        self,
        title: str | None,
        abstract: str | None,
        first_paragraph: str | None = None,
    ) -> str:
        """Generate a 2-sentence summary of a patent."""
        try:
            # Prepare content for summarization
            title_text = title or "No title provided"
            abstract_text = abstract or "No abstract provided"

            # Use first paragraph (usually [0001]) for background context
            background_text = ""
            if first_paragraph:
                # Limit to reasonable length for prompt
                background_text = first_paragraph[:800] + "..." if len(first_paragraph) > 800 else first_paragraph
            else:
                background_text = "No background description provided"

            prompt = SUMMARIZATION_PROMPT.format(
                title=title_text,
                abstract=abstract_text,
                background=background_text,
            )

            LOGGER.info("Generating patent summary", extra={"title": title_text[:50]})

            if self.settings.llm_provider == "ollama":
                return self._summarize_with_ollama(prompt)
            else:
                return self._summarize_with_openai(prompt)

        except Exception as e:
            LOGGER.warning("Failed to generate patent summary", extra={"error": str(e)})
            return self._fallback_summary(title, abstract)

    def _summarize_with_ollama(self, prompt: str) -> str:
        """Generate summary using Ollama/Llama 3.1."""
        try:
            response = ollama.chat(
                model=self.settings.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a technical patent analyzer."},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 100,  # Limit response length
                },
            )

            summary = response["message"]["content"]
            if summary:
                return summary.strip()
            else:
                raise ValueError("Empty response from Ollama")

        except Exception as e:
            LOGGER.warning("Ollama summarization failed", extra={"error": str(e)})
            raise

    def _summarize_with_openai(self, prompt: str) -> str:
        """Generate summary using OpenAI API (fallback)."""
        client = OpenAI(api_key=self.settings.openai_model_id)

        response = client.chat.completions.create(
            model=self.settings.openai_model_id,
            messages=[
                {"role": "system", "content": "You are a technical patent analyzer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        summary = response.choices[0].message.content
        if summary:
            return summary.strip()
        else:
            raise ValueError("Empty response from OpenAI")

    def _fallback_summary(self, title: str | None, abstract: str | None) -> str:
        """Fallback summary when LLM generation fails."""
        if title and abstract:
            # Try to create a simple summary from title + first sentence of abstract
            abstract_first = abstract.split('.')[0] if abstract else ""
            return f"{title}. {abstract_first}."[:200]
        elif title:
            return f"{title}. Technical patent document."
        elif abstract:
            first_sentence = abstract.split('.')[0] if abstract else abstract
            return f"{first_sentence}. Patent document with detailed technical specifications."[:200]
        else:
            return "Technical patent document with detailed specifications and claims."