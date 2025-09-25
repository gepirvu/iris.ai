"""Routes serving the lightweight frontend experience."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["frontend"])

PROJECT_ROOT = Path(__file__).resolve().parents[3].parent
INDEX_HTML = PROJECT_ROOT / "frontend" / "index.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def render_frontend() -> HTMLResponse:
    """Serve the static search UI."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="Frontend assets missing")

    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
