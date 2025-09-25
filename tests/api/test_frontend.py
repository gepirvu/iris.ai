"""Tests for the HTML frontend route."""

from fastapi.testclient import TestClient

from patent_rag_app.api.main import create_app


client = TestClient(create_app())


def test_frontend_route_returns_html() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Patent RAG Demo" in response.text
