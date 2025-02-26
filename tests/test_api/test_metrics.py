"""Tests for the metrics API routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.routes.metrics import router
from src.api.models import MetricsRequest, MetricsResponse, MetricsTimeSeriesResponse
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


# Mock MetricsTracker
class MockMetricsTracker:
    """Mock MetricsTracker for testing."""
    
    async def update_data(self, nodes, edges):
        """Mock update_data method."""
        pass
    
    async def compute_node_count(self):
        """Mock compute_node_count method."""
        return 100
    
    async def compute_edge_count(self):
        """Mock compute_edge_count method."""
        return 200
    
    async def compute_density(self):
        """Mock compute_density method."""
        return 0.1
    
    async def get_history(self):
        """Mock get_history method."""
        return [
            {
                "timestamp": datetime.utcnow(),
                "metrics": {
                    "node_count": 50,
                    "edge_count": 100,
                    "density": 0.05,
                }
            },
            {
                "timestamp": datetime.utcnow(),
                "metrics": {
                    "node_count": 100,
                    "edge_count": 200,
                    "density": 0.1,
                }
            },
        ]


# Mock GraphMetricsTracker
class MockGraphMetricsTracker:
    """Mock GraphMetricsTracker for testing."""
    
    async def update_data(self, nodes, edges):
        """Mock update_data method."""
        pass
    
    async def compute_clustering_coefficient(self):
        """Mock compute_clustering_coefficient method."""
        return 0.3
    
    async def compute_average_path_length(self):
        """Mock compute_average_path_length method."""
        return 2.5
    
    async def compute_diameter(self):
        """Mock compute_diameter method."""
        return 5
    
    async def get_history(self):
        """Mock get_history method."""
        return [
            {
                "timestamp": datetime.utcnow(),
                "metrics": {
                    "clustering_coefficient": 0.2,
                    "average_path_length": 2.0,
                    "diameter": 4,
                }
            },
            {
                "timestamp": datetime.utcnow(),
                "metrics": {
                    "clustering_coefficient": 0.3,
                    "average_path_length": 2.5,
                    "diameter": 5,
                }
            },
        ]


# Mock AdvancedAnalytics
class MockAdvancedAnalytics:
    """Mock AdvancedAnalytics for testing."""
    
    def __init__(self, graph_manager):
        """Initialize mock AdvancedAnalytics."""
        self.graph_manager = graph_manager
    
    async def compute_community_modularity(self):
        """Mock compute_community_modularity method."""
        return 0.7
    
    async def compute_hub_score(self):
        """Mock compute_hub_score method."""
        return 0.8
    
    async def compute_authority_score(self):
        """Mock compute_authority_score method."""
        return 0.9


# Mock GraphManager
class MockGraphManager:
    """Mock GraphManager for testing."""
    
    async def get_concepts(self):
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
    
    async def get_relationships(self):
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


# Test client
@pytest.fixture
def client():
    """Create test client."""
    # Override dependencies
    app.dependency_overrides[has_permission] = mock_has_permission
    
    with patch("src.api.routes.metrics.MetricsTracker", return_value=MockMetricsTracker()):
        with patch("src.api.routes.metrics.GraphMetricsTracker", return_value=MockGraphMetricsTracker()):
            with patch("src.api.routes.metrics.AdvancedAnalytics", return_value=MockAdvancedAnalytics(None)):
                with patch("src.api.routes.metrics.GraphManager", return_value=MockGraphManager()):
                    with TestClient(app) as client:
                        yield client


def test_get_metrics(client):
    """Test get_metrics endpoint."""
    # Test valid data
    metrics_data = {
        "metrics": ["node_count", "edge_count", "density", "clustering_coefficient", "community_modularity"],
        "from_timestamp": "2023-01-01T00:00:00Z",
        "to_timestamp": "2023-12-31T23:59:59Z",
    }
    response = client.post("/metrics", json=metrics_data)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "timestamp" in data
    assert "node_count" in data["metrics"]
    assert "edge_count" in data["metrics"]
    assert "density" in data["metrics"]
    assert "clustering_coefficient" in data["metrics"]
    assert "community_modularity" in data["metrics"]
    assert data["metrics"]["node_count"] == 100
    assert data["metrics"]["edge_count"] == 200
    assert data["metrics"]["density"] == 0.1
    assert data["metrics"]["clustering_coefficient"] == 0.3
    assert data["metrics"]["community_modularity"] == 0.7
    
    # Test invalid data (missing required field)
    invalid_data = {
        "from_timestamp": "2023-01-01T00:00:00Z",
        "to_timestamp": "2023-12-31T23:59:59Z",
    }
    response = client.post("/metrics", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]


def test_get_metrics_time_series(client):
    """Test get_metrics_time_series endpoint."""
    # Test valid data
    metrics_data = {
        "metrics": ["node_count", "edge_count", "density"],
        "from_timestamp": "2023-01-01T00:00:00Z",
        "to_timestamp": "2023-12-31T23:59:59Z",
    }
    response = client.post("/metrics/time-series", json=metrics_data)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3  # One for each metric
    assert data[0]["metric"] in ["node_count", "edge_count", "density"]
    assert data[1]["metric"] in ["node_count", "edge_count", "density"]
    assert data[2]["metric"] in ["node_count", "edge_count", "density"]
    assert data[0]["metric"] != data[1]["metric"] != data[2]["metric"]
    assert len(data[0]["data"]) == 2  # Two data points for each metric
    assert "timestamp" in data[0]["data"][0]
    assert "value" in data[0]["data"][0]
    
    # Test invalid data (missing required field)
    invalid_data = {
        "from_timestamp": "2023-01-01T00:00:00Z",
        "to_timestamp": "2023-12-31T23:59:59Z",
    }
    response = client.post("/metrics/time-series", json=invalid_data)
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]


def test_get_available_metrics(client):
    """Test get_available_metrics endpoint."""
    response = client.get("/metrics/available")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "node_count" in data
    assert "edge_count" in data
    assert "density" in data
    assert "clustering_coefficient" in data
    assert "average_path_length" in data
    assert "diameter" in data
    assert "community_modularity" in data
    assert "hub_score" in data
    assert "authority_score" in data


def test_stream_metrics(client):
    """Test stream_metrics endpoint."""
    # This is a simplified test since we can't easily test streaming responses
    # In a real test, we would use a more sophisticated approach
    
    # Test valid data
    response = client.get("/metrics/stream?metrics=node_count,edge_count&interval=5")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    # Test invalid data (missing required field)
    response = client.get("/metrics/stream?interval=5")
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]
