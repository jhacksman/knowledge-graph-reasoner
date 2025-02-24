"""Tests for graph metrics computation."""
import pytest
import numpy as np

from src.metrics.graph_metrics import GraphMetricsComputer
from src.models.node import Node
from src.models.edge import Edge


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    nodes = [
        Node(
            id=f"node_{i}",
            content=f"Content {i}",
            embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"type": "concept"}
        )
        for i in range(5)
    ]
    
    edges = [
        Edge(
            source="node_0",
            target="node_1",
            type="related",
            metadata={}
        ),
        Edge(
            source="node_1",
            target="node_2",
            type="related",
            metadata={}
        ),
        Edge(
            source="node_2",
            target="node_3",
            type="related",
            metadata={}
        ),
        Edge(
            source="node_3",
            target="node_4",
            type="related",
            metadata={}
        ),
        Edge(
            source="node_0",
            target="node_4",
            type="related",
            metadata={}
        )
    ]
    
    return nodes, edges


def test_build_networkx_graph(sample_graph):
    """Test NetworkX graph construction."""
    nodes, edges = sample_graph
    G = GraphMetricsComputer._build_networkx_graph(nodes, edges)
    
    assert len(G.nodes()) == 5
    assert len(G.edges()) == 5
    assert all(G.nodes[node.id]["content"] == node.content for node in nodes)


def test_compute_avg_path_length(sample_graph):
    """Test average path length computation."""
    nodes, edges = sample_graph
    avg_path = GraphMetricsComputer.compute_avg_path_length(nodes, edges)
    
    assert isinstance(avg_path, float)
    assert avg_path > 0


def test_compute_diameter(sample_graph):
    """Test diameter computation."""
    nodes, edges = sample_graph
    diameter = GraphMetricsComputer.compute_diameter(nodes, edges)
    
    assert isinstance(diameter, int)
    assert diameter > 0


def test_compute_modularity(sample_graph):
    """Test modularity computation."""
    nodes, edges = sample_graph
    modularity = GraphMetricsComputer.compute_modularity(nodes, edges)
    
    assert isinstance(modularity, float)
    assert 0 <= modularity <= 1


def test_compute_bridge_nodes(sample_graph):
    """Test bridge node identification."""
    nodes, edges = sample_graph
    bridge_nodes, ratio = GraphMetricsComputer.compute_bridge_nodes(nodes, edges)
    
    assert isinstance(bridge_nodes, set)
    assert isinstance(ratio, float)
    assert 0 <= ratio <= 1


def test_compute_metrics(sample_graph):
    """Test computation of all metrics."""
    nodes, edges = sample_graph
    metrics = GraphMetricsComputer.compute_metrics(nodes, edges)
    
    assert isinstance(metrics, dict)
    assert "avg_path_length" in metrics
    assert "diameter" in metrics
    assert "modularity" in metrics
    assert "bridge_node_ratio" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_empty_graph():
    """Test metrics computation with empty graph."""
    metrics = GraphMetricsComputer.compute_metrics([], [])
    
    assert metrics["avg_path_length"] == float('inf')
    assert metrics["diameter"] == float('inf')
    assert metrics["modularity"] == 0.0
    assert metrics["bridge_node_ratio"] == 0.0


def test_disconnected_graph():
    """Test metrics computation with disconnected graph."""
    nodes = [
        Node(
            id=f"node_{i}",
            content=f"Content {i}",
            embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"type": "concept"}
        )
        for i in range(4)
    ]
    
    edges = [
        Edge(
            source="node_0",
            target="node_1",
            type="related",
            metadata={}
        ),
        Edge(
            source="node_2",
            target="node_3",
            type="related",
            metadata={}
        )
    ]
    
    metrics = GraphMetricsComputer.compute_metrics(nodes, edges)
    assert isinstance(metrics["avg_path_length"], float)
    assert isinstance(metrics["diameter"], float)
    assert isinstance(metrics["modularity"], float)
    assert isinstance(metrics["bridge_node_ratio"], float)
