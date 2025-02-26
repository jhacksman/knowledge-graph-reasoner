"""Tests for the expansion API routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.routes.expansion import router
from src.api.models import ExpansionConfig, ExpansionStatus, ExpansionEvent
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


# Mock ReasoningPipeline
class MockReasoningPipeline:
    """Mock ReasoningPipeline for testing."""
    
    async def expand_graph(self, seed_concepts, max_iterations=10, expansion_breadth=5, domains=None, checkpoint_interval=None):
        """Mock expand_graph method."""
        return "123"
    
    async def get_expansion_status(self, expansion_id):
        """Mock get_expansion_status method."""
        if expansion_id == "123":
            return {
                "id": "123",
                "config": {
                    "seed_concepts": ["456", "789"],
                    "max_iterations": 10,
                    "expansion_breadth": 5,
                    "domains": ["test"],
                    "checkpoint_interval": 2,
                },
                "current_iteration": 5,
                "total_concepts": 100,
                "total_relationships": 200,
                "status": "running",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        return None
    
    async def pause_expansion(self, expansion_id):
        """Mock pause_expansion method."""
        return True
    
    async def resume_expansion(self, expansion_id):
        """Mock resume_expansion method."""
        return True
    
    async def cancel_expansion(self, expansion_id):
        """Mock cancel_expansion method."""
        return True
    
    async def get_expansion_events(self, expansion_id, limit=10):
        """Mock get_expansion_events method."""
        if expansion_id == "123":
            return [
                {
                    "expansion_id": "123",
                    "event_type": "iteration_complete",
                    "data": {"iteration": 1, "new_concepts": 10},
                    "timestamp": datetime.utcnow(),
                },
                {
                    "expansion_id": "123",
                    "event_type": "iteration_complete",
                    "data": {"iteration": 2, "new_concepts": 15},
                    "timestamp": datetime.utcnow(),
                },
            ]
        return []


# Mock GraphManager
class MockGraphManager:
    """Mock GraphManager for testing."""
    
    async def get_concept(self, concept_id):
        """Mock get_concept method."""
        if concept_id in ["456", "789"]:
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


# Test client
@pytest.fixture
def client():
    """Create test client."""
    # Override dependencies
    app.dependency_overrides[has_permission] = mock_has_permission
    
    with patch("src.api.routes.expansion.ReasoningPipeline", return_value=MockReasoningPipeline()):
        with patch("src.api.routes.expansion.GraphManager", return_value=MockGraphManager()):
            with TestClient(app) as client:
                yield client


def test_start_expansion(client):
    """Test start_expansion endpoint."""
    # Test valid data
    expansion_data = {
        "seed_concepts": ["456", "789"],
        "max_iterations": 10,
        "expansion_breadth": 5,
        "domains": ["test"],
        "checkpoint_interval": 2,
    }
    response = client.post("/expansion", json=expansion_data)
    assert response.status_code == 202
    data = response.json()
    assert data["id"] == "123"
    assert data["status"] == "pending"
    
    # Test invalid data (missing required field)
    invalid_data = {
        "max_iterations": 10,
        "expansion_breadth": 5,
    }
    response = client.post("/expansion", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]
    
    # Test invalid seed concept
    invalid_seed = {
        "seed_concepts": ["999"],  # Non-existent concept
        "max_iterations": 10,
        "expansion_breadth": 5,
    }
    response = client.post("/expansion", json=invalid_seed)
    assert response.status_code == 404
    assert "Seed concept" in response.json()["detail"]


def test_get_expansion_status(client):
    """Test get_expansion_status endpoint."""
    # Test valid expansion ID
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 5,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "running",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/expansion/123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["status"] == "running"
        assert data["current_iteration"] == 5
        assert data["total_concepts"] == 100
        assert data["total_relationships"] == 200
    
    # Test invalid expansion ID
    response = client.get("/expansion/456")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_pause_expansion(client):
    """Test pause_expansion endpoint."""
    # Test valid expansion ID with running status
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 5,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "running",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/expansion/123/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["status"] == "paused"
    
    # Test valid expansion ID with completed status
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 10,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/expansion/123/pause")
        assert response.status_code == 400
        assert "Cannot pause" in response.json()["detail"]
    
    # Test invalid expansion ID
    response = client.post("/expansion/456/pause")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_resume_expansion(client):
    """Test resume_expansion endpoint."""
    # Test valid expansion ID with paused status
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 5,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "paused",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/expansion/123/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["status"] == "running"
    
    # Test valid expansion ID with running status
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 5,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "running",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/expansion/123/resume")
        assert response.status_code == 400
        assert "Cannot resume" in response.json()["detail"]
    
    # Test invalid expansion ID
    response = client.post("/expansion/456/resume")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_cancel_expansion(client):
    """Test cancel_expansion endpoint."""
    # Test valid expansion ID with running status
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 5,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "running",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/expansion/123/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["status"] == "cancelled"
    
    # Test valid expansion ID with completed status
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 10,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.post("/expansion/123/cancel")
        assert response.status_code == 400
        assert "Cannot cancel" in response.json()["detail"]
    
    # Test invalid expansion ID
    response = client.post("/expansion/456/cancel")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_list_expansions(client):
    """Test list_expansions endpoint."""
    # Test with expansions
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 5,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "running",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        },
        "456": {
            "id": "456",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 10,
            "total_concepts": 200,
            "total_relationships": 400,
            "status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/expansion")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] in ["123", "456"]
        assert data[1]["id"] in ["123", "456"]
        assert data[0]["id"] != data[1]["id"]
    
    # Test with no expansions
    with patch("src.api.routes.expansion.EXPANSIONS", {}):
        response = client.get("/expansion")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0


def test_get_expansion_events(client):
    """Test get_expansion_events endpoint."""
    # Test valid expansion ID
    with patch("src.api.routes.expansion.EXPANSION_EVENTS", {
        "123": [
            {
                "expansion_id": "123",
                "event_type": "iteration_complete",
                "data": {"iteration": 1, "new_concepts": 10},
                "timestamp": datetime.utcnow(),
            },
            {
                "expansion_id": "123",
                "event_type": "iteration_complete",
                "data": {"iteration": 2, "new_concepts": 15},
                "timestamp": datetime.utcnow(),
            },
        ]
    }):
        response = client.get("/expansion/123/events")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["expansion_id"] == "123"
        assert data[0]["event_type"] == "iteration_complete"
        assert data[0]["data"]["iteration"] == 1
        assert data[1]["expansion_id"] == "123"
        assert data[1]["event_type"] == "iteration_complete"
        assert data[1]["data"]["iteration"] == 2
    
    # Test invalid expansion ID
    response = client.get("/expansion/456/events")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_stream_expansion_events(client):
    """Test stream_expansion_events endpoint."""
    # This is a simplified test since we can't easily test streaming responses
    # In a real test, we would use a more sophisticated approach
    
    # Test valid expansion ID
    with patch("src.api.routes.expansion.EXPANSIONS", {
        "123": {
            "id": "123",
            "config": {
                "seed_concepts": ["456", "789"],
                "max_iterations": 10,
                "expansion_breadth": 5,
                "domains": ["test"],
                "checkpoint_interval": 2,
            },
            "current_iteration": 5,
            "total_concepts": 100,
            "total_relationships": 200,
            "status": "running",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }):
        response = client.get("/expansion/123/stream")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
    
    # Test invalid expansion ID
    response = client.get("/expansion/456/stream")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
