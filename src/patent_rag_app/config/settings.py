"""Application configuration via Pydantic settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT_DIR / ".env"


class AppSettings(BaseSettings):
    """Global application configuration."""

    environment: str = Field(default="development")
    openai_model_id: str = Field(alias="OPENAI_MODEL_ID")
    llm_provider: str = Field(alias="LLM_PROVIDER", default="ollama")
    ollama_model: str = Field(alias="OLLAMA_MODEL", default="llama3.2")
    ollama_url: str = Field(alias="OLLAMA_URL", default="http://localhost:11434")
    mongodb_uri: str = Field(alias="DATABASE_HOST")
    mongodb_user: str = Field(alias="DATABASE_USER")
    mongodb_password: str = Field(alias="DATABASE_PASSWORD")
    mongo_database: str = Field(default="patent_rag")
    patents_collection: str = Field(default="patents")
    chunks_collection: str = Field(default="chunks")
    qdrant_url: str = Field(alias="QDRANT_CLOUD_URL")
    qdrant_api_key: str = Field(alias="QDRANT_APIKEY")
    use_qdrant_cloud: bool = Field(alias="USE_QDRANT_CLOUD", default=True)
    huggingface_token: str | None = Field(alias="HUGGINGFACE_ACCESS_TOKEN", default=None)
    collection_name: str = Field(alias="COLLECTION_NAME")
    embedding_model: str = Field(alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=16)
    chunk_size: int = Field(alias="CHUNK_SIZE", default=512)
    chunk_overlap: int = Field(alias="CHUNK_OVERLAP", default=50)
    reranker_model: str | None = Field(alias="RERANKER_MODEL", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_k: int = Field(alias="RERANKER_TOP_K", default=25)

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", env_file_encoding="utf-8")

    def snapshot(self) -> dict[str, Any]:
        """Return a sanitized dictionary of public settings."""
        return {
            "environment": self.environment,
            "openai_model_id": self.openai_model_id,
            "mongodb_host": self.sanitize_uri(self.mongodb_uri),
            "qdrant_url": self.sanitize_uri(self.qdrant_url),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    @staticmethod
    def sanitize_uri(uri: str) -> str:
        """Remove credentials from connection URIs for public display."""
        parsed = urlparse(uri)
        netloc = parsed.netloc.split("@")[-1] if parsed.netloc else uri
        return f"{parsed.scheme}://{netloc}" if parsed.scheme else netloc


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load settings once per process."""
    return AppSettings()
