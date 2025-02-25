"""Tests for reasoning pipeline."""
import pytest
import networkx as nx
from unittest.mock import AsyncMock, MagicMock

from typing import List, Optional, Dict, Any

from src.reasoning.pipeline import ReasoningPipeline
from src.reasoning.llm import VeniceLLM
from src.graph.manager import GraphManager
from src.vector_store.base import BaseVectorStore
from src.models.node import Node
from src.models.edge import Edge


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing."""

    async def add_embedding(self, id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Add embedding to store."""
        pass

    async def search(self, embedding: Any, k: int = 5, threshold: float = 0.7):
        """Search for similar embeddings."""
        return []

    async def delete(self, id: str):
        """Delete embedding from store."""
        pass


class MockLLM(VeniceLLM):
    """Mock LLM for testing."""

    async def embed_text(self, text: str) -> Any:
        """Generate embedding for text."""
        return [0.1, 0.2, 0.3]

    async def generate(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> Any:
        """Generate text from messages."""
        return {
            "choices": [
                {
                    "message": {
                        "content": """<entity>test_entity: This is a test entity</entity>
<relationship>test_entity: related_entity: related_to: This is a test relationship</relationship>"""
                    }
                }
            ]
        }


class MockGraphManager(GraphManager):
    """Mock graph manager for testing."""

    def __init__(self):
        """Initialize mock graph manager."""
        self.metrics = MagicMock()
        self.metrics.graph = nx.Graph()
        self.metrics.get_modularity.return_value = 0.5
        self.metrics.get_average_path_length.return_value = 4.5
        self.metrics.get_bridge_nodes.return_value = ["node1", "node2"]
        self.metrics.get_diameter.return_value = 16.0

    async def add_concept(self, content: str, embedding: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add concept to graph."""
        return "concept_id"

    async def get_concept(self, id: str) -> Optional[Node]:
        """Get concept from graph."""
        return Node(id=id, content="Test concept")

    async def add_relationship(self, source_id: str, target_id: str, type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add relationship to graph."""
        return "relationship_id"

    async def get_relationship(self, id: str) -> Optional[Edge]:
        """Get relationship from graph."""
        return Edge(source="source", target="target", type="test")

    async def get_similar_concepts(self, embedding: Any, k: int = 5, threshold: float = 0.7) -> List[Node]:
        """Get similar concepts."""
        return []

    async def get_graph_state(self) -> dict:
        """Get current graph state."""
        return {
            "modularity": 0.5,
            "avg_path_length": 4.5,
            "bridge_nodes": ["node1", "node2"],
            "diameter": 16.0
        }


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    return MockLLM(api_key="test", model="test")


@pytest.fixture
def mock_graph():
    """Create mock graph manager."""
    return MockGraphManager()


@pytest.fixture
def pipeline(mock_llm, mock_graph):
    """Create reasoning pipeline."""
    # Mock metrics for hub formation and community preservation
    mock_graph.metrics = MagicMock()
    mock_graph.metrics.graph = nx.Graph()
    
    return ReasoningPipeline(
        llm=mock_llm,
        graph=mock_graph,
        max_iterations=3,
        stability_window=2,
        enable_hub_formation=True,
        enable_community_preservation=True,
        target_modularity=0.69,
        target_power_law_exponent=3.0
    )


@pytest.mark.asyncio
async def test_init(mock_llm, mock_graph):
    """Test initialization."""
    pipeline = ReasoningPipeline(
        llm=mock_llm,
        graph=mock_graph,
        max_iterations=5,
        stability_window=2
    )
    
    assert pipeline.llm == mock_llm
    assert pipeline.graph == mock_graph
    assert pipeline.max_iterations == 5
    assert pipeline.stability_window == 2


@pytest.mark.asyncio
async def test_check_stability(pipeline):
    """Test stability check."""
    # Not enough history
    assert await pipeline._check_stability() is False
    
    # Add history with unstable path length
    pipeline.metric_history = [
        {"avg_path_length": 3.0, "diameter": 16.0},
        {"avg_path_length": 3.5, "diameter": 16.0},
    ]
    assert await pipeline._check_stability() is False
    
    # Add history with stable metrics
    pipeline.metric_history = [
        {"avg_path_length": 4.5, "diameter": 16.0},
        {"avg_path_length": 4.8, "diameter": 17.0},
    ]
    assert await pipeline._check_stability() is True


@pytest.mark.asyncio
async def test_generate_concepts(pipeline, mock_llm):
    """Test concept generation."""
    concepts = await pipeline._generate_concepts(
        "test concept",
        {"modularity": 0.5, "avg_path_length": 4.5, "bridge_nodes": []},
        {"domain": "test"}
    )
    
    assert len(concepts) == 1
    assert concepts[0]["name"] == "test_entity"
    assert "relationships" in concepts[0]


@pytest.mark.asyncio
async def test_integrate_concepts(pipeline, mock_graph):
    """Test concept integration."""
    # Mock the embed_text method
    pipeline.llm.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    
    # Create test concepts
    concepts = [{
        "name": "test_concept",
        "content": "Test content",
        "metadata": {},
        "relationships": [{
            "source": "test_concept",
            "target": "related_concept",
            "type": "related_to",
            "description": "Test relationship"
        }]
    }, {
        "name": "related_concept",
        "content": "Related content",
        "metadata": {},
        "relationships": []
    }]
    
    # Mock the find_duplicates method
    pipeline.deduplication.find_duplicates = AsyncMock(return_value={})
    
    # Test integration
    await pipeline._integrate_concepts(concepts)
    
    mock_graph.add_concept.assert_called_once()
    assert mock_graph.add_relationship.call_count >= 1


@pytest.mark.asyncio
async def test_apply_self_organization(pipeline, mock_graph):
    """Test self-organization mechanisms."""
    # Mock hub formation methods
    pipeline.hub_formation.identify_potential_hubs = AsyncMock(return_value=["hub1", "hub2"])
    pipeline.hub_formation.strengthen_hub_connections = AsyncMock(return_value={
        "hub_count": 2,
        "strengthened_connections": 5,
        "new_connections": 1
    })
    pipeline.hub_formation.analyze_hub_structure = AsyncMock(return_value={
        "scale_free_properties": {
            "is_scale_free": True,
            "power_law_exponent": 2.8
        }
    })
    
    # Mock community preservation methods
    pipeline.community_preservation.detect_communities = AsyncMock(return_value=[
        {"0", "1", "2"}, {"3", "4", "5"}
    ])
    pipeline.community_preservation.preserve_community_structure = AsyncMock(return_value={
        "preserved_communities": 2,
        "strengthened_connections": 3
    })
    pipeline.community_preservation.optimize_modularity = AsyncMock(return_value={
        "initial_modularity": 0.5,
        "target_modularity": 0.69,
        "final_modularity": 0.65
    })
    pipeline.community_preservation.analyze_community_evolution = AsyncMock(return_value={
        "stability_score": 0.8
    })
    
    # Test with previous communities
    previous_communities = [{"0", "1"}, {"2", "3", "4"}]
    result = await pipeline._apply_self_organization(previous_communities)
    
    # Verify hub formation was called
    pipeline.hub_formation.identify_potential_hubs.assert_called_once()
    pipeline.hub_formation.strengthen_hub_connections.assert_called_once()
    
    # Verify community preservation was called
    pipeline.community_preservation.detect_communities.assert_called_once()
    pipeline.community_preservation.preserve_community_structure.assert_called_once()
    pipeline.community_preservation.optimize_modularity.assert_called_once()
    pipeline.community_preservation.analyze_community_evolution.assert_called_once()
    
    # Verify results
    assert "hub_formation" in result
    assert "community_preservation" in result
    assert "modularity_optimization" in result
    assert "community_evolution" in result
    
    # Test error handling
    pipeline.hub_formation.identify_potential_hubs.side_effect = Exception("Test error")
    result = await pipeline._apply_self_organization(previous_communities)
    assert "error" in result


@pytest.mark.asyncio
async def test_expand_knowledge_with_self_organization(pipeline, mock_llm, mock_graph):
    """Test knowledge expansion with self-organization."""
    # Mock self-organization method
    pipeline._apply_self_organization = AsyncMock(return_value={
        "hub_formation": {"hub_count": 2},
        "community_preservation": {"preserved_communities": 2}
    })
    
    # Mock hub analysis and community metrics for final state
    pipeline.hub_formation.analyze_hub_structure = AsyncMock(return_value={
        "scale_free_properties": {
            "is_scale_free": True,
            "power_law_exponent": 2.8
        }
    })
    
    pipeline.community_preservation.get_community_metrics = AsyncMock(return_value={
        "community_count": 2,
        "modularity": 0.69
    })
    
    # Expand knowledge with self-organization
    final_state = await pipeline.expand_knowledge(
        "test concept",
        context={"domain": "test"},
        self_organization_interval=1
    )
    
    # Verify self-organization was applied
    assert pipeline._apply_self_organization.call_count >= 1
    
    # Verify final state includes self-organization metrics
    assert "hub_analysis" in final_state
    assert "community_metrics" in final_state
    
    # Test with disabled self-organization
    pipeline.enable_hub_formation = False
    pipeline.enable_community_preservation = False
    
    final_state = await pipeline.expand_knowledge(
        "test concept",
        context={"domain": "test"}
    )
    
    # Verify self-organization metrics are not included
    assert "hub_analysis" not in final_state
    assert "community_metrics" not in final_state
