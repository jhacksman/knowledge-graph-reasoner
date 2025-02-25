"""Tests for the queries API routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.routes.queries import router
from src.api.models import Query, QueryStatus, QueryResult
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


# Mock QueryExecutor
class MockQueryExecutor:
    """Mock QueryExecutor for testing."""
    
    async def execute(self, query_text, max_results=10, include_reasoning=False):
        """Mock execute method."""
        return {
            "id": "123",
            "text": query_text,
            "concepts": [
                {
                    "id": "456",
                    "name": "Test Concept",
                    "description": "A test concept",
                    "domain": "test",
                    "attributes": {"key": "value"},
                }
            ],
            "relationships": [
                {
                    "id": "789",
                    "source_id": "456",
                    "target_id": "789",
                    "type": "related_to",
                    "weight": 0.8,
                    "attributes": {"key": "value"},
                }
            ],
            "reasoning": "Test reasoning" if include_reasoning else None,
            "created_at": datetime.utcnow(),
        }


# Test client
@pytest.fixture
def client():
    """Create test client."""
    # Override GraphManager dependency
    app.dependency_overrides[has_permission] = mock_has_permission
    
    with patch("src.api.routes.queries.QueryExecutor", return_value=MockQueryExecutor()):
        with TestClient(app) as client:
            yield client


def test_submit_query(client):
    """Test submit_query endpoint."""
    # Test valid data
    query_data = {
        "text": "Test query",
        "max_results": 10,
        "include_reasoning": True,
    }
    response = client.post("/queries", json=query_data)
    assert response.status_code == 202
    data = response.json()
    assert data["id"] is not None
    assert data["text"] == "Test query"
    assert data["status"] == "pending"
    
    # Test invalid data (missing required field)
    invalid_data = {
        "max_results": 10,
        "include_reasoning": True,
    }
    response = client.post("/queries", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]


def test_get_query_status(client):
    """Test get_query_status endpoint."""
    # Test valid query ID
    with patch("src.api.routes.queries.QUERIES", {
        "123": {
            "id": "123",
            "text": "Test query",
            "max_results": 10,
            "include_reasoning": True,
            "status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/queries/123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["text"] == "Test query"
        assert data["status"] == "completed"
    
    # Test invalid query ID
    response = client.get("/queries/456")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_query_result(client):
    """Test get_query_result endpoint."""
    # Test valid query ID with completed status
    with patch("src.api.routes.queries.QUERIES", {
        "123": {
            "id": "123",
            "text": "Test query",
            "max_results": 10,
            "include_reasoning": True,
            "status": "completed",
            "result": {
                "id": "123",
                "text": "Test query",
                "concepts": [
                    {
                        "id": "456",
                        "name": "Test Concept",
                        "description": "A test concept",
                        "domain": "test",
                        "attributes": {"key": "value"},
                    }
                ],
                "relationships": [
                    {
                        "id": "789",
                        "source_id": "456",
                        "target_id": "789",
                        "type": "related_to",
                        "weight": 0.8,
                        "attributes": {"key": "value"},
                    }
                ],
                "reasoning": "Test reasoning",
                "created_at": datetime.utcnow().isoformat(),
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/queries/123/result")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["text"] == "Test query"
        assert len(data["concepts"]) == 1
        assert data["concepts"][0]["id"] == "456"
        assert len(data["relationships"]) == 1
        assert data["relationships"][0]["id"] == "789"
        assert data["reasoning"] == "Test reasoning"
    
    # Test valid query ID with pending status
    with patch("src.api.routes.queries.QUERIES", {
        "123": {
            "id": "123",
            "text": "Test query",
            "max_results": 10,
            "include_reasoning": True,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/queries/123/result")
        assert response.status_code == 202
        data = response.json()
        assert data["id"] == "123"
        assert data["status"] == "pending"
    
    # Test valid query ID with failed status
    with patch("src.api.routes.queries.QUERIES", {
        "123": {
            "id": "123",
            "text": "Test query",
            "max_results": 10,
            "include_reasoning": True,
            "status": "failed",
            "error": "Test error",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/queries/123/result")
        assert response.status_code == 500
        data = response.json()
        assert "Test error" in data["detail"]
    
    # Test invalid query ID
    response = client.get("/queries/456/result")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_list_queries(client):
    """Test list_queries endpoint."""
    # Test with queries
    with patch("src.api.routes.queries.QUERIES", {
        "123": {
            "id": "123",
            "text": "Test query 1",
            "max_results": 10,
            "include_reasoning": True,
            "status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        },
        "456": {
            "id": "456",
            "text": "Test query 2",
            "max_results": 5,
            "include_reasoning": False,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/queries")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] in ["123", "456"]
        assert data[1]["id"] in ["123", "456"]
        assert data[0]["id"] != data[1]["id"]
    
    # Test with no queries
    with patch("src.api.routes.queries.QUERIES", {}):
        response = client.get("/queries")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0


def test_cancel_query(client):
    """Test cancel_query endpoint."""
    # Test valid query ID with pending status
    with patch("src.api.routes.queries.QUERIES", {
        "123": {
            "id": "123",
            "text": "Test query",
            "max_results": 10,
            "include_reasoning": True,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/queries/123/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["status"] == "cancelled"
    
    # Test valid query ID with completed status
    with patch("src.api.routes.queries.QUERIES", {
        "123": {
            "id": "123",
            "text": "Test query",
            "max_results": 10,
            "include_reasoning": True,
            "status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/queries/123/cancel")
        assert response.status_code == 400
        assert "Cannot cancel" in response.json()["detail"]
    
    # Test invalid query ID
    response = client.post("/queries/456/cancel")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
