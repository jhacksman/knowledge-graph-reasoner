"""Tests for reasoning pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from typing import List, Dict, Any, Optional, AsyncIterator

from src.reasoning.pipeline import ReasoningPipeline
from src.reasoning.llm import VeniceLLM
from src.graph.manager import GraphManager
from src.vector_store.base import BaseVectorStore
from src.models.node import Node
from src.models.edge import Edge


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


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    llm = MagicMock(spec=VeniceLLM)
    llm.embed_text = AsyncMock(return_value=np.random.rand(1536))
    llm.generate = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "concept 1\nconcept 2\nconcept 3"
            }
        }]
    })
    return llm


@pytest.fixture
def mock_graph():
    """Create mock graph manager."""
    graph = MagicMock(spec=GraphManager)
    graph.add_concept = AsyncMock(return_value="test_id")
    graph.add_relationship = AsyncMock()
    graph.get_similar_concepts = AsyncMock(return_value=[
        Node(id="similar1", content="Similar 1"),
        Node(id="similar2", content="Similar 2")
    ])
    graph.get_graph_state = AsyncMock(return_value={
        "modularity": 0.75,
        "avg_path_length": 4.8,
        "bridge_nodes": ["test1", "test2"],
        "diameter": 16.5
    })
    return graph


@pytest.fixture
def pipeline(mock_llm, mock_graph):
    """Create reasoning pipeline."""
    return ReasoningPipeline(
        llm=mock_llm,
        graph=mock_graph,
        max_iterations=3,
        stability_window=2
    )


@pytest.mark.asyncio
async def test_expand_knowledge(pipeline, mock_llm, mock_graph):
    """Test knowledge expansion."""
    # Expand knowledge
    final_state = await pipeline.expand_knowledge(
        "test concept",
        context={"domain": "test"}
    )
    
    # Verify LLM calls
    assert mock_llm.embed_text.call_count >= 1
    assert mock_llm.generate.call_count >= 1
    
    # Verify graph operations
    assert mock_graph.add_concept.call_count >= 1
    assert mock_graph.add_relationship.call_count >= 1
    
    # Verify final state
    assert isinstance(final_state, dict)
    assert "modularity" in final_state
    assert "avg_path_length" in final_state
    assert "bridge_nodes" in final_state


@pytest.mark.asyncio
async def test_stability_check(pipeline):
    """Test stability checking."""
    # Setup unstable metrics
    pipeline.metric_history = [{
        "avg_path_length": 3.0,
        "diameter": 10.0
    } for _ in range(5)]
    
    # Should not be stable
    assert not await pipeline._check_stability()
    
    # Setup stable metrics
    pipeline.metric_history = [{
        "avg_path_length": 4.8,
        "diameter": 17.0
    } for _ in range(5)]
    
    # Should be stable
    assert await pipeline._check_stability()


@pytest.mark.asyncio
async def test_concept_generation(pipeline, mock_llm):
    """Test concept generation."""
    concepts = await pipeline._generate_concepts(
        "test concept",
        {
            "modularity": 0.75,
            "avg_path_length": 4.8,
            "bridge_nodes": ["test1", "test2"],
            "diameter": 16.5
        },
        {"domain": "test"}
    )
    
    assert len(concepts) == 3
    assert all("content" in c for c in concepts)
    assert all("metadata" in c for c in concepts)
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_concept_integration(pipeline, mock_graph):
    """Test concept integration."""
    concepts = [{
        "content": "test concept",
        "metadata": {"key": "value"}
    }]
    
    await pipeline._integrate_concepts(concepts)
    
    mock_graph.add_concept.assert_called_once()
    assert mock_graph.add_relationship.call_count >= 1
