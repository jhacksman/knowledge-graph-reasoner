"""Tests for graph metrics computation."""
import pytest
import networkx as nx
from typing import List, Dict, Any

from src.metrics.metrics import GraphMetrics


@pytest.fixture
def sample_graph() -> nx.Graph:
    """Create a sample test graph."""
    G = nx.Graph()
    
    # Add nodes
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    G.add_nodes_from(nodes)
    
    # Add edges to create communities
    edges = [
        ("A", "B"), ("B", "C"), ("C", "A"),  # Community 1
        ("D", "E"), ("E", "F"), ("F", "D"),  # Community 2
        ("G", "H"),                          # Community 3
        ("C", "D"), ("F", "G")              # Bridge edges
    ]
    G.add_edges_from(edges)
    
    return G


@pytest.fixture
def metrics() -> GraphMetrics:
    """Create metrics instance."""
    return GraphMetrics()


@pytest.mark.asyncio
async def test_modularity_computation(metrics: GraphMetrics, sample_graph: nx.Graph):
    """Test modularity score computation."""
    # Update graph
    metrics.graph = sample_graph
    
    # Compute modularity
    modularity = await metrics.compute_modularity()
    
    assert isinstance(modularity, float)
    assert 0 <= modularity <= 1
    assert modularity > 0.3  # Should have clear community structure


@pytest.mark.asyncio
async def test_path_length_computation(metrics: GraphMetrics, sample_graph: nx.Graph):
    """Test average path length computation."""
    # Update graph
    metrics.graph = sample_graph
    
    # Compute average path length
    avg_path = await metrics.compute_avg_path_length()
    
    assert isinstance(avg_path, float)
    assert avg_path > 0
    assert 2.0 <= avg_path <= 3.0  # Expected range for sample graph


@pytest.mark.asyncio
async def test_diameter_computation(metrics: GraphMetrics, sample_graph: nx.Graph):
    """Test graph diameter computation."""
    # Update graph
    metrics.graph = sample_graph
    
    # Compute diameter
    diameter = await metrics.compute_diameter()
    
    assert isinstance(diameter, float)
    assert diameter > 0
    assert 3 <= diameter <= 5  # Expected range for sample graph


@pytest.mark.asyncio
async def test_bridge_node_detection(metrics: GraphMetrics, sample_graph: nx.Graph):
    """Test bridge node identification."""
    # Update graph
    metrics.graph = sample_graph
    
    # Find bridge nodes
    bridge_nodes = await metrics.find_bridge_nodes()
    
    assert isinstance(bridge_nodes, list)
    assert len(bridge_nodes) >= 2
    assert "C" in bridge_nodes  # Should be a bridge
    assert "F" in bridge_nodes  # Should be a bridge


@pytest.mark.asyncio
async def test_hub_centrality(metrics: GraphMetrics, sample_graph: nx.Graph):
    """Test hub centrality computation."""
    # Update graph
    metrics.graph = sample_graph
    
    # Compute hub centrality
    centrality = await metrics.compute_hub_centrality()
    
    assert isinstance(centrality, dict)
    assert len(centrality) == len(sample_graph)
    assert all(isinstance(score, float) for score in centrality.values())
    assert all(0 <= score <= 1 for score in centrality.values())


@pytest.mark.asyncio
async def test_path_length_distribution(metrics: GraphMetrics, sample_graph: nx.Graph):
    """Test path length distribution computation."""
    # Update graph
    metrics.graph = sample_graph
    
    # Compute distribution
    distribution = await metrics.compute_path_length_distribution()
    
    assert isinstance(distribution, dict)
    assert len(distribution) > 0
    assert all(isinstance(length, int) for length in distribution.keys())
    assert all(isinstance(count, int) for count in distribution.values())
    assert 1 in distribution  # Should have direct connections
    assert 2 in distribution  # Should have 2-hop paths


@pytest.mark.asyncio
async def test_stability_check(metrics: GraphMetrics):
    """Test graph stability checking."""
    # Create history with stable metrics
    metrics.history = [
        {
            "modularity": 0.75,
            "avg_path_length": 4.8,
            "diameter": 17.0,
            "bridge_nodes": ["node1", "node2"]
        }
        for _ in range(5)
    ]
    
    # Check stability
    is_stable = await metrics.check_stability()
    assert is_stable
    
    # Add unstable metrics
    metrics.history.append({
        "modularity": 0.6,  # Significant drop
        "avg_path_length": 3.0,  # Out of range
        "diameter": 15.0,  # Out of range
        "bridge_nodes": ["node1"]
    })
    
    is_stable = await metrics.check_stability()
    assert not is_stable


@pytest.mark.asyncio
async def test_graph_updates(metrics: GraphMetrics):
    """Test graph structure updates."""
    # Define test data
    nodes = ["1", "2", "3"]
    edges = [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"}
    ]
    
    # Update graph
    await metrics.update_graph(nodes, edges, track_metrics=True)
    
    # Verify graph structure
    assert len(metrics.graph) == 3
    assert len(metrics.graph.edges) == 2
    assert len(metrics.history) == 1
    
    # Verify metrics were tracked
    latest = metrics.history[-1]
    assert "modularity" in latest
    assert "avg_path_length" in latest
    assert "diameter" in latest
    assert "bridge_nodes" in latest


@pytest.mark.asyncio
async def test_empty_graph_handling(metrics: GraphMetrics):
    """Test handling of empty graphs."""
    # Empty graph
    modularity = await metrics.compute_modularity()
    assert modularity == 0.0
    
    avg_path = await metrics.compute_avg_path_length()
    assert avg_path == 0.0
    
    diameter = await metrics.compute_diameter()
    assert diameter == 0.0
    
    bridge_nodes = await metrics.find_bridge_nodes()
    assert len(bridge_nodes) == 0
    
    centrality = await metrics.compute_hub_centrality()
    assert len(centrality) == 0
    
    distribution = await metrics.compute_path_length_distribution()
    assert len(distribution) == 0


@pytest.mark.asyncio
async def test_single_node_handling(metrics: GraphMetrics):
    """Test handling of single-node graphs."""
    # Add single node
    metrics.graph.add_node("A")
    
    modularity = await metrics.compute_modularity()
    assert modularity == 0.0
    
    avg_path = await metrics.compute_avg_path_length()
    assert avg_path == 0.0
    
    diameter = await metrics.compute_diameter()
    assert diameter == 0.0
    
    bridge_nodes = await metrics.find_bridge_nodes()
    assert len(bridge_nodes) == 0
    
    centrality = await metrics.compute_hub_centrality()
    assert len(centrality) == 1
    assert "A" in centrality
