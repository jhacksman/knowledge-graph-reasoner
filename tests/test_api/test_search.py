"""Tests for the search API routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.routes.search import router
from src.api.models import SearchRequest, SearchResponse, SearchResult
from src.api.auth import get_api_key, has_permission


# Mock API key dependency
async def mock_get_api_key():
    """Mock API key dependency."""
    return MagicMock(
        key="test-key",
        name="Test Key",
        role="admin",
        created_at="2023-01-01T00:00:00Z",
        last_used_at=None,
        usage_count=0,
    )


# Mock permission dependency
def mock_has_permission(permission):
    """Mock permission dependency."""
    async def dependency(api_key=None):
        return None
    return dependency


# Create test app
app = FastAPI()
app.include_router(router)
app.dependency_overrides[get_api_key] = mock_get_api_key


# Mock VeniceLLM
class MockVeniceLLM:
    """Mock VeniceLLM for testing."""
    
    async def embed_text(self, text):
        """Mock embed_text method."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]


# Mock GraphManager
class MockGraphManager:
    """Mock GraphManager for testing."""
    
    async def search_concepts_by_embedding(self, embedding, limit=10, threshold=0.7, domains=None):
        """Mock search_concepts_by_embedding method."""
        return [
            (
                MagicMock(
                    id="123",
                    name="Test Concept 1",
                    description="A test concept",
                    domain="test",
                    attributes={"key": "value"},
                    embedding=[0.1, 0.2, 0.3],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                ),
                0.9,
            ),
            (
                MagicMock(
                    id="456",
                    name="Test Concept 2",
                    description="Another test concept",
                    domain="test",
                    attributes={"key": "value"},
                    embedding=[0.2, 0.3, 0.4],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                ),
                0.8,
            ),
        ]
    
    async def get_concept(self, concept_id):
        """Mock get_concept method."""
        if concept_id == "123":
            return MagicMock(
                id="123",
                name="Test Concept 1",
                description="A test concept",
                domain="test",
                attributes={"key": "value"},
                embedding=[0.1, 0.2, 0.3],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        elif concept_id == "456":
            return MagicMock(
                id="456",
                name="Test Concept 2",
                description="Another test concept",
                domain="test",
                attributes={"key": "value"},
                embedding=[0.2, 0.3, 0.4],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        return None


# Test client
@pytest.fixture
def client():
    """Create test client."""
    # Override dependencies
    app.dependency_overrides[has_permission] = mock_has_permission
    
    with patch("src.api.routes.search.VeniceLLM", return_value=MockVeniceLLM()):
        with patch("src.api.routes.search.GraphManager", return_value=MockGraphManager()):
            with TestClient(app) as client:
                yield client


def test_search_concepts(client):
    """Test search_concepts endpoint."""
    # Test valid data
    search_data = {
        "query": "test query",
        "limit": 10,
        "threshold": 0.7,
        "domains": ["test"],
    }
    response = client.post("/search", json=search_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert "total" in data
    assert data["query"] == "test query"
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["concept"]["id"] == "123"
    assert data["results"][0]["concept"]["name"] == "Test Concept 1"
    assert data["results"][0]["similarity"] == 0.9
    assert data["results"][1]["concept"]["id"] == "456"
    assert data["results"][1]["concept"]["name"] == "Test Concept 2"
    assert data["results"][1]["similarity"] == 0.8
    
    # Test invalid data (missing required field)
    invalid_data = {
        "limit": 10,
        "threshold": 0.7,
    }
    response = client.post("/search", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]


def test_search_by_text(client):
    """Test search_by_text endpoint."""
    # Test valid data
    response = client.get("/search/by-text/test%20query?limit=10&threshold=0.7&domains=test")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert "total" in data
    assert data["query"] == "test query"
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["concept"]["id"] == "123"
    assert data["results"][0]["concept"]["name"] == "Test Concept 1"
    assert data["results"][0]["similarity"] == 0.9
    assert data["results"][1]["concept"]["id"] == "456"
    assert data["results"][1]["concept"]["name"] == "Test Concept 2"
    assert data["results"][1]["similarity"] == 0.8


def test_search_by_concept(client):
    """Test search_by_concept endpoint."""
    # Test valid concept ID
    response = client.get("/search/by-concept/123?limit=10&threshold=0.7&domains=test")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert "total" in data
    assert "Similar to concept: Test Concept 1" in data["query"]
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["concept"]["id"] == "123"
    assert data["results"][0]["concept"]["name"] == "Test Concept 1"
    assert data["results"][0]["similarity"] == 0.9
    assert data["results"][1]["concept"]["id"] == "456"
    assert data["results"][1]["concept"]["name"] == "Test Concept 2"
    assert data["results"][1]["similarity"] == 0.8
    
    # Test invalid concept ID
    response = client.get("/search/by-concept/999?limit=10&threshold=0.7&domains=test")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
