"""Tests for the relationships API routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.routes.relationships import router
from src.api.models import Relationship, RelationshipCreate, RelationshipUpdate, RelationshipList
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
    
    async def get_relationships(self, **kwargs):
        """Mock get_relationships method."""
        return [
            MagicMock(
                id="789",
                source_id="123",
                target_id="456",
                type="related_to",
                weight=0.8,
                attributes={"key": "value"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        ]
    
    async def count_relationships(self, **kwargs):
        """Mock count_relationships method."""
        return 1
    
    async def get_relationship(self, relationship_id):
        """Mock get_relationship method."""
        if relationship_id == "789":
            return MagicMock(
                id="789",
                source_id="123",
                target_id="456",
                type="related_to",
                weight=0.8,
                attributes={"key": "value"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        return None
    
    async def get_concept(self, concept_id):
        """Mock get_concept method."""
        if concept_id in ["123", "456"]:
            return MagicMock(
                id=concept_id,
                name=f"Concept {concept_id}",
                description=f"Description for {concept_id}",
                domain="test",
                attributes={"key": "value"},
                embedding=[0.1, 0.2, 0.3],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        return None
    
    async def create_relationship(self, **kwargs):
        """Mock create_relationship method."""
        return MagicMock(
            id="789",
            source_id=kwargs.get("source_id"),
            target_id=kwargs.get("target_id"),
            type=kwargs.get("type"),
            weight=kwargs.get("weight"),
            attributes=kwargs.get("attributes"),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    async def update_relationship(self, **kwargs):
        """Mock update_relationship method."""
        return MagicMock(
            id=kwargs.get("relationship_id"),
            source_id="123",
            target_id="456",
            type=kwargs.get("type") or "related_to",
            weight=kwargs.get("weight") or 0.8,
            attributes=kwargs.get("attributes") or {"key": "value"},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    async def delete_relationship(self, relationship_id):
        """Mock delete_relationship method."""
        return None


# Test client
@pytest.fixture
def client():
    """Create test client."""
    # Override GraphManager dependency
    app.dependency_overrides[has_permission] = mock_has_permission
    
    with patch("src.api.routes.relationships.GraphManager", return_value=MockGraphManager()):
        with TestClient(app) as client:
            yield client


def test_list_relationships(client):
    """Test list_relationships endpoint."""
    response = client.get("/relationships")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == "789"
    assert data["items"][0]["source_id"] == "123"
    assert data["items"][0]["target_id"] == "456"


def test_get_relationship(client):
    """Test get_relationship endpoint."""
    # Test valid relationship ID
    response = client.get("/relationships/789")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "789"
    assert data["source_id"] == "123"
    assert data["target_id"] == "456"
    
    # Test invalid relationship ID
    response = client.get("/relationships/999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_create_relationship(client):
    """Test create_relationship endpoint."""
    # Test valid data
    relationship_data = {
        "source_id": "123",
        "target_id": "456",
        "type": "related_to",
        "weight": 0.8,
        "attributes": {"key": "value"},
    }
    response = client.post("/relationships", json=relationship_data)
    assert response.status_code == 201
    data = response.json()
    assert data["source_id"] == "123"
    assert data["target_id"] == "456"
    assert data["type"] == "related_to"
    
    # Test invalid data (missing required field)
    invalid_data = {
        "source_id": "123",
        "type": "related_to",
    }
    response = client.post("/relationships", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]
    
    # Test invalid source concept
    invalid_source = {
        "source_id": "999",  # Non-existent concept
        "target_id": "456",
        "type": "related_to",
    }
    response = client.post("/relationships", json=invalid_source)
    assert response.status_code == 404
    assert "Source concept" in response.json()["detail"]
    
    # Test invalid target concept
    invalid_target = {
        "source_id": "123",
        "target_id": "999",  # Non-existent concept
        "type": "related_to",
    }
    response = client.post("/relationships", json=invalid_target)
    assert response.status_code == 404
    assert "Target concept" in response.json()["detail"]


def test_update_relationship(client):
    """Test update_relationship endpoint."""
    # Test valid update
    update_data = {
        "type": "updated_relation",
        "weight": 0.9,
    }
    response = client.put("/relationships/789", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "789"
    assert data["type"] == "updated_relation"
    assert data["weight"] == 0.9
    
    # Test invalid relationship ID
    response = client.put("/relationships/999", json=update_data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_relationship(client):
    """Test delete_relationship endpoint."""
    # Test valid relationship ID
    response = client.delete("/relationships/789")
    assert response.status_code == 204
    
    # Test invalid relationship ID
    response = client.delete("/relationships/999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_create_relationships_batch(client):
    """Test create_relationships_batch endpoint."""
    # Test valid data
    relationships_data = [
        {
            "source_id": "123",
            "target_id": "456",
            "type": "related_to",
            "weight": 0.8,
            "attributes": {"key": "value1"},
        },
        {
            "source_id": "123",
            "target_id": "456",
            "type": "similar_to",
            "weight": 0.7,
            "attributes": {"key": "value2"},
        },
    ]
    response = client.post("/relationships/batch", json=relationships_data)
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 2
    assert data[0]["type"] == "related_to"
    assert data[1]["type"] == "similar_to"
    
    # Test invalid data (missing required field)
    invalid_data = [
        {
            "source_id": "123",
            "target_id": "456",
            "type": "related_to",
        },
        {
            "source_id": "123",
            "type": "similar_to",  # Missing target_id
        },
    ]
    response = client.post("/relationships/batch", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]
    
    # Test invalid source concept
    invalid_source = [
        {
            "source_id": "999",  # Non-existent concept
            "target_id": "456",
            "type": "related_to",
        },
    ]
    response = client.post("/relationships/batch", json=invalid_source)
    assert response.status_code == 404
    assert "Source concept" in response.json()["detail"]
