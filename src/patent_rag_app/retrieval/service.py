"""Retrieval services for patent chunk search."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple

from qdrant_client.http import models as qmodels

from patent_rag_app.config.logging import get_logger
from patent_rag_app.config.settings import AppSettings, get_settings
from patent_rag_app.db.chunk_repository import ChunkRepository
from patent_rag_app.ingestion.schemas import ChunkDocument
from patent_rag_app.llm.embeddings import EmbeddingEncoder
from patent_rag_app.llm.reranker import CrossEncoderReranker
from patent_rag_app.retrieval.qdrant_store import QdrantStore

PARAGRAPH_PATTERN = re.compile(r"(?:para[_\s-]?|paragraph\s*)(0\d{3})|\b(0\d{3})\b|\[(0\d{3})\]")
CLAIM_PATTERN = re.compile(r"\bclaim\s+(\d{1,4})\b", re.IGNORECASE)

FIELD_PRIORITIES = {
    "45": 0,
    "43": 1,
}

DEFAULT_MIN_VECTOR_SCORE = 0.25
LEXICAL_TOP_K = 40
SEMANTIC_TOP_K = 60
RRF_K = 60

SYNONYM_EXPANSIONS = {
    "iron loss": ["hysteresis loss", "eddy current loss"],
    "resistivity": ["electrical resistance"],
    "non-oriented electrical steel": ["NOES"],
}

PATENT_ID_ALIASES = {
    "EP2679695A1": "EP1577413_A1",
    "EP1577413A1": "EP1577413_A1",
}

LOGGER = get_logger(__name__)

FIELD_SYNONYMS = {
    "11": ["publication number", "patent number", "pub number"],
    "12": ["document kind"],
    "21": ["application number"],
    "22": ["filing date", "date of filing"],
    "30": ["priority", "priority data"],
    "43": [
        "publication date",
        "date of publication",
        "date publication",
        "publication date of application",
        "date of publication of application",
        "application publication date",
        "publication of the application",
        "(43)",
    ],
    "45": [
        "grant publication",
        "grant date",
        "date of grant",
        "publication of the grant",
        "publication and mention of the grant",
        "date of publication and mention of the grant",
        "grant of the patent",
        "(45)",
    ],
    "51": ["classification", "ipc"],
    "54": ["title"],
    "56": ["citation", "references"],
    "57": ["abstract"],
    "71": ["applicant"],
    "72": ["inventor"],
    "73": ["assignee", "owner"],
    "74": ["representative", "agent"],
    "84": ["designated states", "states"],
    "86": ["international application"],
    "87": ["international publication"],
}


@dataclass
class ChunkSearchResult:
    chunk_id: str
    chunk_key: str
    patent_id: str
    section_label: str
    text: str
    score: float


class PatentRetrievalService:
    """Search patent chunks stored in Qdrant."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        encoder: EmbeddingEncoder | None = None,
        store: QdrantStore | None = None,
        chunk_repository: ChunkRepository | None = None,
        reranker: CrossEncoderReranker | None = None,
        min_vector_score: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.encoder = encoder or EmbeddingEncoder(self.settings)
        self.store = store or QdrantStore(settings=self.settings)
        self.chunk_repository = chunk_repository or ChunkRepository(settings=self.settings)
        self.reranker = reranker
        if self.reranker is None and self.settings.reranker_model:
            try:
                self.reranker = CrossEncoderReranker(self.settings)
            except Exception as exc:  # pragma: no cover - defensive fallback
                LOGGER.warning(
                    "Reranker unavailable",
                    extra={"error": str(exc), "model": self.settings.reranker_model},
                )
                self.reranker = None
        self.min_vector_score = (
            DEFAULT_MIN_VECTOR_SCORE if min_vector_score is None else float(min_vector_score)
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        patent_id: str | None = None,
    ) -> List[ChunkSearchResult]:
        original_query = query
        rewritten_query, alias_patent_id = self._rewrite_query(query)
        if alias_patent_id and not patent_id:
            patent_id = alias_patent_id

        direct_hits = self._direct_match(rewritten_query, patent_id, top_k)
        if direct_hits is not None:
            return direct_hits

        vector = self.encoder.encode([rewritten_query])[0].tolist()
        fetch_k = SEMANTIC_TOP_K if patent_id is None else max(SEMANTIC_TOP_K, top_k * 3)

        query_filter = None
        if patent_id is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="patent_id",
                        match=qmodels.MatchValue(value=patent_id),
                    )
                ]
            )

        hits = self.store.search(vector, top_k=fetch_k, query_filter=query_filter)

        semantic_hits = [
            self._hit_to_result(hit)
            for hit in hits
            if (hit.score or 0.0) >= self.min_vector_score
        ]

        lexical_hits = self._lexical_search(rewritten_query, patent_id, limit=LEXICAL_TOP_K)

        fused = self._fuse_results(semantic_hits, lexical_hits)
        if not fused:
            return []

        reranked = self._rerank_results(original_query, fused)
        return reranked[:top_k]

    def _direct_match(
        self,
        query: str,
        patent_id: str | None,
        top_k: int,
    ) -> List[ChunkSearchResult] | None:
        if not patent_id:
            return None

        normalized = query.lower()
        if "how many" in normalized and "claim" in normalized:
            total = self.chunk_repository.count_claims(patent_id)
            summary = f"{patent_id} has {total} claim{'s' if total != 1 else ''}."
            return [
                ChunkSearchResult(
                    chunk_id=f"{patent_id}_claim_count",
                    chunk_key=f"count_{patent_id}_claims",
                    patent_id=patent_id,
                    section_label="Claim count",
                    text=summary,
                    score=1.0,
                )
            ]

        if "claims" in normalized:
            claims = self.chunk_repository.fetch_claims(patent_id, limit=top_k)
            if claims:
                return [self._chunk_to_result(claim) for claim in claims]

        table_match = re.search(r"table\s+(\d+)", normalized)
        if table_match:
            table_id = table_match.group(1).zfill(4)
            chunk = self.chunk_repository.fetch_by_section(patent_id, f"table_{table_id}")
            if chunk:
                return [self._chunk_to_result(chunk)]

        section_code = self._match_paragraph(normalized)
        if section_code:
            chunk = self.chunk_repository.fetch_by_section(patent_id, section_code)
            if chunk:
                return [self._chunk_to_result(chunk)]

        claim_code = self._match_claim(normalized)
        if claim_code:
            chunk = self.chunk_repository.fetch_by_section(patent_id, claim_code)
            if chunk:
                return [self._chunk_to_result(chunk)]

        field_code = self._match_front_field(normalized)
        if field_code:
            chunk_key = f"front_{patent_id}_{field_code}"
            chunk = self.chunk_repository.fetch_by_chunk_key(chunk_key)
            if chunk:
                return [self._chunk_to_result(chunk)]

        return None

    @staticmethod
    def _match_paragraph(text: str) -> str | None:
        match = PARAGRAPH_PATTERN.search(text)
        if not match:
            return None
        for part in match.groups():
            if part:
                return f"para_{part}"
        return None

    @staticmethod
    def _match_claim(text: str) -> str | None:
        match = CLAIM_PATTERN.search(text)
        if not match:
            return None
        number = match.group(1).strip()
        padded = number.zfill(4)
        return f"claim_{padded}"

    @staticmethod
    def _match_front_field(text: str) -> str | None:
        matches: list[Tuple[str, str]] = []
        for code, synonyms in FIELD_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in text:
                    matches.append((code, synonym))
                    break
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0][0]
        matches.sort(key=lambda item: (FIELD_PRIORITIES.get(item[0], 100), -len(item[1])))
        return matches[0][0]

    @staticmethod
    def _chunk_to_result(chunk: ChunkDocument) -> ChunkSearchResult:
        return ChunkSearchResult(
            chunk_id=chunk.chunk_id,
            chunk_key=chunk.chunk_key,
            patent_id=chunk.patent_id,
            section_label=chunk.section_label,
            text=chunk.text,
            score=1.0,
        )

    @staticmethod
    def _hit_to_result(hit) -> ChunkSearchResult:
        payload = hit.payload or {}
        return ChunkSearchResult(
            chunk_id=payload.get("chunk_id", ""),
            chunk_key=payload.get("chunk_key", ""),
            patent_id=payload.get("patent_id", ""),
            section_label=payload.get("section_label", ""),
            text=payload.get("text", ""),
            score=hit.score or 0.0,
        )

    def _lexical_search(
        self,
        query: str,
        patent_id: str | None,
        *,
        limit: int,
    ) -> List[ChunkSearchResult]:
        hits = self.chunk_repository.text_search(query, patent_id=patent_id, limit=limit)
        results: List[ChunkSearchResult] = []
        for score, chunk in hits:
            result = self._chunk_to_result(chunk)
            result.score = float(score)
            results.append(result)
        return results

    def _fuse_results(
        self,
        semantic: List[ChunkSearchResult],
        lexical: List[ChunkSearchResult],
    ) -> List[ChunkSearchResult]:
        if not semantic and not lexical:
            return []

        combined: Dict[str, Tuple[ChunkSearchResult, float]] = {}

        def add_source(source: Iterable[ChunkSearchResult]) -> None:
            for rank, result in enumerate(source, start=1):
                score = 1.0 / (RRF_K + rank)
                existing = combined.get(result.chunk_id)
                if existing is None:
                    combined[result.chunk_id] = (result, score)
                else:
                    combined[result.chunk_id] = (existing[0], existing[1] + score)

        add_source(semantic)
        add_source(lexical)

        ranked = sorted(
            combined.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [item[0] for item in ranked]

    def _rerank_results(
        self,
        query: str,
        results: List[ChunkSearchResult],
    ) -> List[ChunkSearchResult]:
        if not self.reranker or not results:
            return results

        pool_size = min(len(results), self.settings.reranker_top_k)
        pool = results[:pool_size]
        scores = self.reranker.score(query, [item.text for item in pool])

        scored = sorted(
            zip(pool, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        reranked = [item[0] for item in scored]
        if pool_size < len(results):
            reranked.extend(results[pool_size:])
        return reranked

    def _rewrite_query(self, query: str) -> tuple[str, str | None]:
        normalized = query.lower()
        expansions: list[str] = []
        for phrase, synonyms in SYNONYM_EXPANSIONS.items():
            if phrase in normalized:
                expansions.extend(synonyms)

        alias_patent_id = self._detect_patent_alias(query)

        if expansions:
            query = " ".join([query] + expansions)
        return query, alias_patent_id

    def _detect_patent_alias(self, query: str) -> str | None:
        matches = re.findall(r"EP[\s\d]+A1", query, flags=re.IGNORECASE)
        for match in matches:
            normalized = re.sub(r"[^A-Z0-9]", "", match.upper())
            alias = PATENT_ID_ALIASES.get(normalized)
            if alias:
                return alias
            auto = re.fullmatch(r"EP(\d+)A1", normalized)
            if auto:
                digits = auto.group(1)
                return f"EP{digits}_A1"
        return None





