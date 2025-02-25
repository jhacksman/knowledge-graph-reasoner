"""Tests for the concepts API routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.routes.concepts import router
from src.api.models import Concept, ConceptCreate, ConceptUpdate, ConceptList
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


# Mock GraphManager
class MockGraphManager:
    """Mock GraphManager for testing."""
    
    async def get_concepts(self, **kwargs):
        """Mock get_concepts method."""
        return [
            MagicMock(
                id="123",
                name="Test Concept",
                description="A test concept",
                domain="test",
                attributes={"key": "value"},
                embedding=[0.1, 0.2, 0.3],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        ]
    
    async def count_concepts(self, **kwargs):
        """Mock count_concepts method."""
        return 1
    
    async def get_concept(self, concept_id):
        """Mock get_concept method."""
        if concept_id == "123":
            return MagicMock(
                id="123",
                name="Test Concept",
                description="A test concept",
                domain="test",
                attributes={"key": "value"},
                embedding=[0.1, 0.2, 0.3],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        return None
    
    async def create_concept(self, **kwargs):
        """Mock create_concept method."""
        return MagicMock(
            id="123",
            name=kwargs.get("name"),
            description=kwargs.get("description"),
            domain=kwargs.get("domain"),
            attributes=kwargs.get("attributes"),
            embedding=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    async def update_concept(self, **kwargs):
        """Mock update_concept method."""
        return MagicMock(
            id=kwargs.get("concept_id"),
            name=kwargs.get("name") or "Test Concept",
            description=kwargs.get("description") or "A test concept",
            domain=kwargs.get("domain") or "test",
            attributes=kwargs.get("attributes") or {"key": "value"},
            embedding=[0.1, 0.2, 0.3],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    async def delete_concept(self, concept_id):
        """Mock delete_concept method."""
        return None


# Test client
@pytest.fixture
def client():
    """Create test client."""
    # Override GraphManager dependency
    app.dependency_overrides[has_permission] = mock_has_permission
    
    with patch("src.api.routes.concepts.GraphManager", return_value=MockGraphManager()):
        with TestClient(app) as client:
            yield client


def test_list_concepts(client):
    """Test list_concepts endpoint."""
    response = client.get("/concepts")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == "123"
    assert data["items"][0]["name"] == "Test Concept"


def test_get_concept(client):
    """Test get_concept endpoint."""
    # Test valid concept ID
    response = client.get("/concepts/123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "123"
    assert data["name"] == "Test Concept"
    
    # Test invalid concept ID
    response = client.get("/concepts/456")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_create_concept(client):
    """Test create_concept endpoint."""
    # Test valid data
    concept_data = {
        "name": "New Concept",
        "description": "A new concept",
        "domain": "new",
        "attributes": {"key": "new_value"},
    }
    response = client.post("/concepts", json=concept_data)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "New Concept"
    assert data["description"] == "A new concept"
    assert data["domain"] == "new"
    assert data["attributes"] == {"key": "new_value"}
    
    # Test invalid data (missing required field)
    invalid_data = {
        "description": "A new concept",
        "domain": "new",
    }
    response = client.post("/concepts", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]


def test_update_concept(client):
    """Test update_concept endpoint."""
    # Test valid update
    update_data = {
        "name": "Updated Concept",
        "description": "An updated concept",
    }
    response = client.put("/concepts/123", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "123"
    assert data["name"] == "Updated Concept"
    assert data["description"] == "An updated concept"
    
    # Test invalid concept ID
    response = client.put("/concepts/456", json=update_data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_concept(client):
    """Test delete_concept endpoint."""
    # Test valid concept ID
    response = client.delete("/concepts/123")
    assert response.status_code == 204
    
    # Test invalid concept ID
    response = client.delete("/concepts/456")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_create_concepts_batch(client):
    """Test create_concepts_batch endpoint."""
    # Test valid data
    concepts_data = [
        {
            "name": "Concept 1",
            "description": "First concept",
            "domain": "batch",
            "attributes": {"key": "value1"},
        },
        {
            "name": "Concept 2",
            "description": "Second concept",
            "domain": "batch",
            "attributes": {"key": "value2"},
        },
    ]
    response = client.post("/concepts/batch", json=concepts_data)
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "Concept 1"
    assert data[1]["name"] == "Concept 2"
    
    # Test invalid data (missing required field)
    invalid_data = [
        {
            "name": "Concept 1",
            "description": "First concept",
            "domain": "batch",
        },
        {
            "description": "Second concept",  # Missing name
            "domain": "batch",
        },
    ]
    response = client.post("/concepts/batch", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]
