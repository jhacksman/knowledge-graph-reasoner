"""Tests for graph manager."""
import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio
import numpy as np
from typing import List, Optional, AsyncIterator

from src.graph.manager import GraphManager
from src.vector_store.base import BaseVectorStore
from src.models.node import Node
from src.models.edge import Edge
from src.metrics.graph_metrics import GraphMetrics


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing."""
    
    async def initialize(self) -> None:
        pass
    
    async def add_node(self, node: Node) -> str:
        return node.id
    
    async def add_edge(self, edge: Edge) -> None:
        pass
    
    async def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.5,
        deduplicate: bool = True
    ) -> List[Node]:
        return [
            Node(id="test1", content="Test content 1"),
            Node(id="test2", content="Test content 2")
        ]
    
    async def get_node(self, node_id: str) -> Optional[Node]:
        return Node(id=node_id, content=f"Test content for {node_id}")
    
    async def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None
    ) -> AsyncIterator[Edge]:
        edges = [
            Edge(source="test1", target="test2", type="related"),
            Edge(source="test2", target="test3", type="similar")
        ]
        for edge in edges:
            yield edge
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def get_all_nodes(self) -> AsyncIterator[Node]:
        """Get all nodes implementation for MockVectorStore."""
        nodes = [
            Node(id="test1", content="Test content 1"),
            Node(id="test2", content="Test content 2")
        ]
        for node in nodes:
            yield node
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def get_all_edges(self) -> AsyncIterator[Edge]:
        """Get all edges implementation for MockVectorStore."""
        edges = [
            Edge(source="test1", target="test2", type="related"),
            Edge(source="test2", target="test3", type="similar")
        ]
        for edge in edges:
            yield edge
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def update_node(self, node: Node) -> None:
        """Update node implementation for MockVectorStore."""
        # This is a mock method, so no actual implementation needed
        pass


@pytest.fixture
def mock_metrics():
    """Create mock metrics."""
    metrics = MagicMock(spec=GraphMetrics)
    metrics.compute_modularity = AsyncMock(return_value=0.75)
    metrics.compute_avg_path_length = AsyncMock(return_value=4.8)
    metrics.find_bridge_nodes = AsyncMock(return_value=["test1", "test2"])
    metrics.compute_diameter = AsyncMock(return_value=16.5)
    return metrics


@pytest.fixture
def graph_manager(mock_metrics):
    """Create graph manager with mocks."""
    return GraphManager(MockVectorStore(), metrics=mock_metrics)


@pytest.mark.asyncio
async def test_add_concept(graph_manager):
    """Test adding a concept."""
    embedding = np.random.rand(1536)
    concept_id = await graph_manager.add_concept(
        "test concept",
        embedding,
        metadata={"key": "value"}
    )
    
    assert isinstance(concept_id, str)
    assert len(concept_id) > 0


@pytest.mark.asyncio
async def test_add_relationship(graph_manager):
    """Test adding a relationship."""
    await graph_manager.add_relationship(
        "source_id",
        "target_id",
        "test_type",
        metadata={"key": "value"}
    )


@pytest.mark.asyncio
async def test_get_similar_concepts(graph_manager):
    """Test finding similar concepts."""
    embedding = np.random.rand(1536)
    nodes = await graph_manager.get_similar_concepts(
        embedding,
        k=5,
        threshold=0.5
    )
    
    assert len(nodes) == 2
    assert all(isinstance(node, Node) for node in nodes)


@pytest.mark.asyncio
async def test_get_concept(graph_manager):
    """Test getting a concept by ID."""
    node = await graph_manager.get_concept("test_id")
    
    assert node is not None
    assert node.id == "test_id"
    assert "Test content for test_id" in node.content


@pytest.mark.asyncio
async def test_get_relationships(graph_manager):
    """Test getting relationships."""
    edges = await graph_manager.get_relationships(
        source_id="test1",
        target_id="test2",
        relationship_type="related"
    )
    
    assert len(edges) == 2
    assert all(isinstance(edge, Edge) for edge in edges)


@pytest.mark.asyncio
async def test_get_graph_state(graph_manager, mock_metrics):
    """Test getting graph state."""
    state = await graph_manager.get_graph_state()
    
    assert state["modularity"] == 0.75
    assert state["avg_path_length"] == 4.8
    assert len(state["bridge_nodes"]) == 2
    assert state["diameter"] == 16.5
    
    mock_metrics.compute_modularity.assert_called_once()
    mock_metrics.compute_avg_path_length.assert_called_once()
    mock_metrics.find_bridge_nodes.assert_called_once()
    mock_metrics.compute_diameter.assert_called_once()
