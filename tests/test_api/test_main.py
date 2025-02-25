"""Tests for the FastAPI application setup."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app


# Test client
@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as client:
        yield client


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "documentation" in data
    assert "timestamp" in data


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "uptime" in data


def test_docs_endpoint(client):
    """Test docs endpoint."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "swagger" in response.text.lower()


def test_openapi_schema(client):
    """Test OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "components" in data
    assert "schemas" in data
    
    # Check if all routes are included in the schema
    paths = data["paths"]
    assert "/concepts" in paths
    assert "/relationships" in paths
    assert "/queries" in paths
    assert "/expansion" in paths
    assert "/metrics" in paths
    assert "/search" in paths


def test_exception_handlers():
    """Test exception handlers."""
    with TestClient(app) as client:
        # Test HTTP exception handler
        with patch("src.api.main.get_api_key", side_effect=Exception("Test error")):
            response = client.get("/concepts")
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Test error" in data["detail"]
            assert "error_code" in data
            assert "timestamp" in data
            assert "path" in data
