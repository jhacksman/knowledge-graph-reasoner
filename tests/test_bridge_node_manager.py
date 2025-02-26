"""Tests for bridge node manager."""
import pytest
import networkx as nx
import numpy as np
from unittest.mock import patch, MagicMock

from src.graph.bridge_node_manager import BridgeNodeManager


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(range(20))
    
    # Add edges to create a structure with clear communities
    # Community 1: 0-4
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(i, j)
    
    # Community 2: 5-9
    for i in range(5, 10):
        for j in range(i + 1, 10):
            G.add_edge(i, j)
    
    # Community 3: 10-14
    for i in range(10, 15):
        for j in range(i + 1, 15):
            G.add_edge(i, j)
    
    # Community 4: 15-19
    for i in range(15, 20):
        for j in range(i + 1, 20):
            G.add_edge(i, j)
    
    # Add bridge edges
    G.add_edge(2, 7)  # Bridge between community 1 and 2
    G.add_edge(8, 12)  # Bridge between community 2 and 3
    G.add_edge(13, 17)  # Bridge between community 3 and 4
    G.add_edge(4, 16)  # Bridge between community 1 and 4
    
    return G


@pytest.fixture
def domain_mapping():
    """Create a sample domain mapping."""
    return {
        0: "physics",
        1: "physics",
        2: "physics",
        3: "physics",
        4: "physics",
        5: "chemistry",
        6: "chemistry",
        7: "chemistry",
        8: "chemistry",
        9: "chemistry",
        10: "biology",
        11: "biology",
        12: "biology",
        13: "biology",
        14: "biology",
        15: "computer_science",
        16: "computer_science",
        17: "computer_science",
        18: "computer_science",
        19: "computer_science"
    }


@pytest.fixture
def manager():
    """Create a BridgeNodeManager instance."""
    return BridgeNodeManager()


@pytest.fixture
def manager_with_graph(sample_graph):
    """Create a BridgeNodeManager instance with a sample graph."""
    manager = BridgeNodeManager(sample_graph)
    return manager


@pytest.fixture
def manager_with_graph_and_domains(sample_graph, domain_mapping):
    """Create a BridgeNodeManager instance with a sample graph and domain mapping."""
    manager = BridgeNodeManager(sample_graph)
    manager.set_domain_mapping(domain_mapping)
    return manager


async def test_init():
    """Test initialization."""
    manager = BridgeNodeManager()
    assert isinstance(manager.graph, nx.Graph)
    assert len(manager.bridge_history) == 0
    assert len(manager.domain_mapping) == 0
    
    # Test with graph
    G = nx.Graph()
    G.add_node(1)
    manager = BridgeNodeManager(G)
    assert manager.graph == G


async def test_set_graph():
    """Test setting graph."""
    manager = BridgeNodeManager()
    G = nx.Graph()
    G.add_node(1)
    
    manager.set_graph(G)
    assert manager.graph == G


async def test_set_domain_mapping():
    """Test setting domain mapping."""
    manager = BridgeNodeManager()
    domain_mapping = {1: "physics", 2: "chemistry"}
    
    manager.set_domain_mapping(domain_mapping)
    assert manager.domain_mapping == domain_mapping


async def test_identify_bridge_nodes(manager_with_graph):
    """Test identifying bridge nodes."""
    bridge_nodes = await manager_with_graph.identify_bridge_nodes()
    assert len(bridge_nodes) > 0
    
    # Bridge nodes should include our manually created bridges
    bridge_set = set(bridge_nodes)
    assert 2 in bridge_set or 7 in bridge_set  # Bridge between community 1 and 2
    assert 8 in bridge_set or 12 in bridge_set  # Bridge between community 2 and 3
    assert 13 in bridge_set or 17 in bridge_set  # Bridge between community 3 and 4
    assert 4 in bridge_set or 16 in bridge_set  # Bridge between community 1 and 4
    
    # Test with empty graph
    manager = BridgeNodeManager()
    bridge_nodes = await manager.identify_bridge_nodes()
    assert len(bridge_nodes) == 0
    
    # Test with single node
    G = nx.Graph()
    G.add_node(1)
    manager.set_graph(G)
    bridge_nodes = await manager.identify_bridge_nodes()
    assert len(bridge_nodes) == 0


async def test_compute_bridge_node_metrics(manager_with_graph):
    """Test computing bridge node metrics."""
    bridge_nodes = await manager_with_graph.identify_bridge_nodes()
    metrics = await manager_with_graph.compute_bridge_node_metrics(bridge_nodes)
    
    assert len(metrics) > 0
    
    # Check metrics structure
    for node, node_metrics in metrics.items():
        assert "betweenness" in node_metrics
        assert "eigenvector" in node_metrics
        assert "degree" in node_metrics
        assert "influence" in node_metrics
        assert "connected_communities" in node_metrics
        assert "timestamp" in node_metrics
    
    # Test with empty bridge nodes
    metrics = await manager_with_graph.compute_bridge_node_metrics([])
    assert len(metrics) == 0
    
    # Test with empty graph
    manager = BridgeNodeManager()
    metrics = await manager.compute_bridge_node_metrics([1, 2, 3])
    assert len(metrics) == 0


async def test_track_bridge_nodes(manager_with_graph):
    """Test tracking bridge nodes."""
    metrics = await manager_with_graph.track_bridge_nodes()
    assert len(metrics) > 0
    
    # Check that history is updated
    assert len(manager_with_graph.bridge_history) > 0
    
    # Track again to ensure history grows
    metrics2 = await manager_with_graph.track_bridge_nodes()
    
    # Check that history has multiple entries for at least one node
    has_multiple_entries = False
    for node, history in manager_with_graph.bridge_history.items():
        if len(history) > 1:
            has_multiple_entries = True
            break
    
    assert has_multiple_entries
    
    # Test with empty graph
    manager = BridgeNodeManager()
    metrics = await manager.track_bridge_nodes()
    assert len(metrics) == 0


async def test_get_persistent_bridge_nodes(manager_with_graph):
    """Test getting persistent bridge nodes."""
    # Track multiple times to build history
    for _ in range(5):
        await manager_with_graph.track_bridge_nodes()
    
    persistent_nodes = await manager_with_graph.get_persistent_bridge_nodes()
    
    # Should have some persistent nodes
    assert len(persistent_nodes) > 0
    
    # Test with empty history
    manager = BridgeNodeManager()
    persistent_nodes = await manager.get_persistent_bridge_nodes()
    assert len(persistent_nodes) == 0


async def test_get_high_influence_bridge_nodes(manager_with_graph):
    """Test getting high influence bridge nodes."""
    # Track to build history
    await manager_with_graph.track_bridge_nodes()
    
    # Temporarily lower threshold to ensure we get some high influence nodes
    original_threshold = manager_with_graph.influence_threshold
    manager_with_graph.influence_threshold = 0.1
    
    high_influence_nodes = await manager_with_graph.get_high_influence_bridge_nodes()
    
    # Should have some high influence nodes
    assert len(high_influence_nodes) > 0
    
    # Restore threshold
    manager_with_graph.influence_threshold = original_threshold
    
    # Test with empty history
    manager = BridgeNodeManager()
    high_influence_nodes = await manager.get_high_influence_bridge_nodes()
    assert len(high_influence_nodes) == 0


async def test_get_cross_domain_bridge_nodes(manager_with_graph_and_domains):
    """Test getting cross-domain bridge nodes."""
    # Track to build history
    await manager_with_graph_and_domains.track_bridge_nodes()
    
    cross_domain_nodes = await manager_with_graph_and_domains.get_cross_domain_bridge_nodes()
    
    # Should have some cross-domain nodes
    assert len(cross_domain_nodes) > 0
    
    # Test with no domain mapping
    manager = BridgeNodeManager(manager_with_graph_and_domains.graph)
    await manager.track_bridge_nodes()
    cross_domain_nodes = await manager.get_cross_domain_bridge_nodes()
    assert len(cross_domain_nodes) == 0


async def test_analyze_bridge_node_evolution(manager_with_graph):
    """Test analyzing bridge node evolution."""
    # Track multiple times to build history
    for _ in range(3):
        await manager_with_graph.track_bridge_nodes()
    
    # Get a bridge node
    bridge_nodes = await manager_with_graph.identify_bridge_nodes()
    node_id = bridge_nodes[0]
    
    analysis = await manager_with_graph.analyze_bridge_node_evolution(node_id)
    
    # Check analysis structure
    assert "node_id" in analysis
    assert "persistence" in analysis
    assert "stability" in analysis
    assert "influence_trend" in analysis
    assert "community_connections_trend" in analysis
    
    # Test with non-existent node
    analysis = await manager_with_graph.analyze_bridge_node_evolution("non_existent")
    assert analysis["persistence"] == 0
    assert analysis["influence_trend"] == "unknown"


async def test_get_domain_bridge_analysis(manager_with_graph_and_domains):
    """Test getting domain bridge analysis."""
    analysis = await manager_with_graph_and_domains.get_domain_bridge_analysis()
    
    # Check analysis structure
    assert "domains" in analysis
    assert "domain_connections" in analysis
    assert "bridge_distribution" in analysis
    
    # Should have our domains
    assert "physics" in analysis["domains"]
    assert "chemistry" in analysis["domains"]
    assert "biology" in analysis["domains"]
    assert "computer_science" in analysis["domains"]
    
    # Test with no domain mapping
    manager = BridgeNodeManager(manager_with_graph_and_domains.graph)
    analysis = await manager.get_domain_bridge_analysis()
    assert len(analysis["domains"]) == 0


async def test_recommend_new_bridge_nodes(manager_with_graph):
    """Test recommending new bridge nodes."""
    recommendations = await manager_with_graph.recommend_new_bridge_nodes()
    
    # Should have some recommendations
    assert len(recommendations) > 0
    
    # Check recommendation structure
    for rec in recommendations:
        assert "source" in rec
        assert "target" in rec
        assert "source_community" in rec
        assert "target_community" in rec
        assert "score" in rec
    
    # Test with empty graph
    manager = BridgeNodeManager()
    recommendations = await manager.recommend_new_bridge_nodes()
    assert len(recommendations) == 0


async def test_get_bridge_node_summary(manager_with_graph_and_domains):
    """Test getting bridge node summary."""
    # Track to build history
    await manager_with_graph_and_domains.track_bridge_nodes()
    
    summary = await manager_with_graph_and_domains.get_bridge_node_summary()
    
    # Check summary structure
    assert "total_bridge_nodes" in summary
    assert "persistent_bridge_nodes" in summary
    assert "high_influence_bridge_nodes" in summary
    assert "cross_domain_bridge_nodes" in summary
    assert "bridge_nodes" in summary
    assert "domain_analysis" in summary
    assert "recommendations" in summary
    
    # Test with empty graph
    manager = BridgeNodeManager()
    summary = await manager.get_bridge_node_summary()
    assert summary["total_bridge_nodes"] == 0


async def test_error_handling():
    """Test error handling in methods."""
    manager = BridgeNodeManager()
    
    # Create a graph that will cause errors
    G = MagicMock()
    G.nodes.side_effect = Exception("Test exception")
    manager.set_graph(G)
    
    # Test error handling in various methods
    bridge_nodes = await manager.identify_bridge_nodes()
    assert len(bridge_nodes) == 0
    
    metrics = await manager.compute_bridge_node_metrics([1, 2, 3])
    assert len(metrics) == 0
    
    metrics = await manager.track_bridge_nodes()
    assert len(metrics) == 0
    
    persistent_nodes = await manager.get_persistent_bridge_nodes()
    assert len(persistent_nodes) == 0
    
    high_influence_nodes = await manager.get_high_influence_bridge_nodes()
    assert len(high_influence_nodes) == 0
    
    cross_domain_nodes = await manager.get_cross_domain_bridge_nodes()
    assert len(cross_domain_nodes) == 0
    
    analysis = await manager.analyze_bridge_node_evolution(1)
    assert analysis["persistence"] == 0
    
    domain_analysis = await manager.get_domain_bridge_analysis()
    assert len(domain_analysis["domains"]) == 0
    
    recommendations = await manager.recommend_new_bridge_nodes()
    assert len(recommendations) == 0
    
    summary = await manager.get_bridge_node_summary()
    assert summary["total_bridge_nodes"] == 0
