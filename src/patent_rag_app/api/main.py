"""FastAPI application setup."""

from fastapi import FastAPI

from patent_rag_app.api.routers import frontend, health, patents, search
from patent_rag_app.config.logging import configure_logging
from patent_rag_app.config.settings import AppSettings, get_settings

configure_logging()


def create_app() -> FastAPI:
    """Application factory to wire routes and dependencies."""
    settings: AppSettings = get_settings()

    app = FastAPI(
        title="Patent RAG Service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(frontend.router)
    app.include_router(health.router)
    app.include_router(patents.router)
    app.include_router(search.router)

    @app.get("/config", include_in_schema=False)
    def show_runtime_configuration() -> dict[str, str | int | bool]:
        """Return non-sensitive runtime settings for smoke testing."""
        return settings.snapshot()

    return app


app = create_app()
