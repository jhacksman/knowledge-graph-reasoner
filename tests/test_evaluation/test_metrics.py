"""Tests for evaluation metrics module."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.node import Node
from src.models.edge import Edge
from src.evaluation.metrics import GraphEvaluator


@pytest.fixture
def mock_graph_manager():
    """Create a mock graph manager."""
    manager = AsyncMock()
    
    # Mock get_graph_state
    manager.get_graph_state.return_value = {
        "timestamp": 1234567890,
        "node_count": 10,
        "edge_count": 15,
        "modularity": 0.75,
        "avg_path_length": 2.5,
        "diameter": 5,
        "communities": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9", "10"]],
        "bridge_nodes": ["3", "6"],
        "centrality": {
            "1": 0.1, "2": 0.2, "3": 0.8, "4": 0.3, 
            "5": 0.2, "6": 0.7, "7": 0.4, "8": 0.3,
            "9": 0.2, "10": 0.1
        }
    }
    
    # Mock get_concept
    async def mock_get_concept(node_id):
        return Node(
            id=node_id,
            content=f"Content for node {node_id}",
            metadata={"embedding": np.random.rand(10).tolist()}
        )
    
    manager.get_concept = mock_get_concept
    
    # Mock get_all_concepts
    async def mock_get_all_concepts():
        for i in range(1, 11):
            yield Node(
                id=str(i),
                content=f"Content for node {i}",
                metadata={"embedding": np.random.rand(10).tolist()}
            )
    
    manager.get_all_concepts = mock_get_all_concepts
    
    # Mock get_all_relationships
    async def mock_get_all_relationships():
        edges = [
            ("1", "2", "related_to"),
            ("2", "3", "causes"),
            ("3", "4", "part_of"),
            ("4", "5", "related_to"),
            ("5", "6", "causes"),
            ("6", "7", "part_of"),
            ("7", "8", "related_to"),
            ("8", "9", "causes"),
            ("9", "10", "part_of"),
            ("10", "1", "related_to"),
            ("1", "5", "related_to"),
            ("2", "6", "related_to"),
            ("3", "7", "related_to"),
            ("4", "8", "related_to"),
            ("5", "9", "related_to")
        ]
        
        for source, target, rel_type in edges:
            yield Edge(
                source=source,
                target=target,
                type=rel_type,
                metadata={"confidence": 0.9}
            )
    
    manager.get_all_relationships = mock_get_all_relationships
    
    # Mock get_similar_concepts
    async def mock_get_similar_concepts(embedding, k=5, threshold=0.0):
        return [
            Node(
                id=str(i),
                content=f"Similar content {i}",
                metadata={"embedding": np.random.rand(10).tolist()}
            )
            for i in range(1, k+1)
        ]
    
    manager.get_similar_concepts = mock_get_similar_concepts
    
    return manager


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = AsyncMock()
    
    # Mock generate
    async def mock_generate(messages):
        return {
            "choices": [
                {
                    "message": {
                        "content": "0.8"
                    }
                }
            ]
        }
    
    llm.generate = mock_generate
    
    return llm


@pytest.mark.asyncio
async def test_compute_semantic_coherence(mock_graph_manager):
    """Test computing semantic coherence."""
    evaluator = GraphEvaluator(mock_graph_manager)
    
    # Create test community
    community = [
        Node(
            id="1",
            content="Test node 1",
            metadata={"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        ),
        Node(
            id="2",
            content="Test node 2",
            metadata={"embedding": [0.15, 0.25, 0.35, 0.45, 0.55]}
        )
    ]
    
    # Compute coherence
    coherence = await evaluator.compute_semantic_coherence(community)
    
    # Check result
    assert isinstance(coherence, float)
    assert 0.0 <= coherence <= 1.0


@pytest.mark.asyncio
async def test_validate_relationship(mock_graph_manager, mock_llm):
    """Test validating relationship."""
    evaluator = GraphEvaluator(mock_graph_manager, llm=mock_llm)
    
    # Create test edge and nodes
    edge = Edge(
        source="1",
        target="2",
        type="related_to",
        metadata={}
    )
    
    source_node = Node(
        id="1",
        content="Source node content",
        metadata={}
    )
    
    target_node = Node(
        id="2",
        content="Target node content",
        metadata={}
    )
    
    # Validate relationship
    validity = await evaluator.validate_relationship(edge, source_node, target_node)
    
    # Check result
    assert isinstance(validity, float)
    assert 0.0 <= validity <= 1.0


@pytest.mark.asyncio
async def test_compute_domain_coverage(mock_graph_manager):
    """Test computing domain coverage."""
    # Create evaluator with domains
    evaluator = GraphEvaluator(
        mock_graph_manager,
        domains=["science", "technology", "humanities"]
    )
    
    # Compute domain coverage
    coverage = await evaluator.compute_domain_coverage()
    
    # Check result
    assert isinstance(coverage, dict)
    assert all(domain in coverage for domain in ["science", "technology", "humanities"])
    assert all(isinstance(score, float) for score in coverage.values())
    assert all(0.0 <= score <= 1.0 for score in coverage.values())


@pytest.mark.asyncio
async def test_compute_interdisciplinary_metrics(mock_graph_manager):
    """Test computing interdisciplinary metrics."""
    evaluator = GraphEvaluator(mock_graph_manager)
    
    # Compute metrics
    metrics = await evaluator.compute_interdisciplinary_metrics()
    
    # Check result
    assert isinstance(metrics, dict)
    assert "cross_community_edges" in metrics
    assert "interdisciplinary_ratio" in metrics
    assert "bridge_node_centrality" in metrics
    assert all(isinstance(value, (int, float)) for value in metrics.values())


@pytest.mark.asyncio
async def test_compute_novelty_score(mock_graph_manager):
    """Test computing novelty score."""
    evaluator = GraphEvaluator(mock_graph_manager)
    
    # Create test node
    node = Node(
        id="test",
        content="Test node content",
        metadata={"embedding": np.random.rand(10).tolist()}
    )
    
    # Compute novelty
    novelty = await evaluator.compute_novelty_score(node)
    
    # Check result
    assert isinstance(novelty, float)
    assert 0.0 <= novelty <= 1.0


@pytest.mark.asyncio
async def test_evaluate(mock_graph_manager):
    """Test comprehensive evaluation."""
    evaluator = GraphEvaluator(mock_graph_manager)
    
    # Perform evaluation
    results = await evaluator.evaluate(iteration=5)
    
    # Check result
    assert isinstance(results, dict)
    assert "iteration" in results
    assert results["iteration"] == 5
    assert "node_count" in results
    assert "edge_count" in results
    assert "modularity" in results


@pytest.mark.asyncio
async def test_compare_iterations(mock_graph_manager):
    """Test comparing iterations."""
    evaluator = GraphEvaluator(mock_graph_manager)
    
    # Add history entries
    evaluator.history = [
        {
            "iteration": 1,
            "node_count": 5,
            "edge_count": 7,
            "modularity": 0.6,
            "avg_path_length": 2.0,
            "domain_coverage": {
                "science": 0.3,
                "technology": 0.4
            }
        },
        {
            "iteration": 2,
            "node_count": 8,
            "edge_count": 12,
            "modularity": 0.7,
            "avg_path_length": 2.2,
            "domain_coverage": {
                "science": 0.4,
                "technology": 0.5
            }
        }
    ]
    
    # Compare iterations
    comparison = await evaluator.compare_iterations(1, 2)
    
    # Check result
    assert isinstance(comparison, dict)
    assert "iterations" in comparison
    assert comparison["iterations"] == [1, 2]
    assert "metrics" in comparison
    assert "domain_coverage" in comparison


@pytest.mark.asyncio
async def test_detect_anomalies(mock_graph_manager):
    """Test anomaly detection."""
    evaluator = GraphEvaluator(mock_graph_manager)
    
    # Add history entries with consistent values
    evaluator.history = [
        {
            "iteration": 1,
            "node_count": 5,
            "edge_count": 7,
            "modularity": 0.6
        },
        {
            "iteration": 2,
            "node_count": 6,
            "edge_count": 8,
            "modularity": 0.62
        },
        {
            "iteration": 3,
            "node_count": 7,
            "edge_count": 9,
            "modularity": 0.64
        },
        # Add anomaly
        {
            "iteration": 4,
            "node_count": 20,  # Sudden jump
            "edge_count": 10,
            "modularity": 0.66
        }
    ]
    
    # Detect anomalies
    anomalies = await evaluator.detect_anomalies(window_size=3, threshold=2.0)
    
    # Check result
    assert isinstance(anomalies, list)
    assert len(anomalies) > 0
    assert all("metric" in anomaly for anomaly in anomalies)
    assert all("value" in anomaly for anomaly in anomalies)
    assert all("z_score" in anomaly for anomaly in anomalies)
    
    # Check if node_count anomaly was detected
    node_count_anomaly = next((a for a in anomalies if a["metric"] == "node_count"), None)
    assert node_count_anomaly is not None
    assert node_count_anomaly["value"] == 20
