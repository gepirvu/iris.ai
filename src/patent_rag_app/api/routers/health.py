"""Health check endpoints."""

from fastapi import APIRouter

from patent_rag_app.config.settings import get_settings


router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness probe")
async def healthcheck() -> dict[str, str]:
    """Return basic app health information."""
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}

