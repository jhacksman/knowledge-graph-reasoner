"""Tests for bridge node integration."""
import pytest
import networkx as nx
import numpy as np
from unittest.mock import patch, MagicMock

from src.graph.bridge_node_integration import BridgeNodeIntegration


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
def integration():
    """Create a BridgeNodeIntegration instance."""
    return BridgeNodeIntegration()


@pytest.fixture
def integration_with_graph(sample_graph):
    """Create a BridgeNodeIntegration instance with a sample graph."""
    integration = BridgeNodeIntegration(sample_graph)
    return integration


@pytest.fixture
def integration_with_graph_and_domains(sample_graph, domain_mapping):
    """Create a BridgeNodeIntegration instance with a sample graph and domain mapping."""
    integration = BridgeNodeIntegration(sample_graph, domain_mapping)
    return integration


async def test_init():
    """Test initialization."""
    integration = BridgeNodeIntegration()
    assert isinstance(integration.manager.graph, nx.Graph)
    assert len(integration.manager.bridge_history) == 0
    assert len(integration.manager.domain_mapping) == 0
    
    # Test with graph
    G = nx.Graph()
    G.add_node(1)
    integration = BridgeNodeIntegration(G)
    assert integration.manager.graph == G
    
    # Test with domain mapping
    domain_mapping = {1: "physics", 2: "chemistry"}
    integration = BridgeNodeIntegration(G, domain_mapping)
    assert integration.manager.domain_mapping == domain_mapping
    
    # Test with thresholds
    integration = BridgeNodeIntegration(
        G,
        persistence_threshold=5,
        influence_threshold=0.7,
        cross_domain_threshold=3
    )
    assert integration.manager.persistence_threshold == 5
    assert integration.manager.influence_threshold == 0.7
    assert integration.manager.cross_domain_threshold == 3


async def test_set_graph():
    """Test setting graph."""
    integration = BridgeNodeIntegration()
    G = nx.Graph()
    G.add_node(1)
    
    integration.set_graph(G)
    assert integration.manager.graph == G
    assert integration.metrics.graph == G


async def test_set_domain_mapping():
    """Test setting domain mapping."""
    integration = BridgeNodeIntegration()
    domain_mapping = {1: "physics", 2: "chemistry"}
    
    integration.set_domain_mapping(domain_mapping)
    assert integration.manager.domain_mapping == domain_mapping


async def test_update_bridge_nodes(integration_with_graph):
    """Test updating bridge nodes."""
    summary = await integration_with_graph.update_bridge_nodes()
    
    # Check summary structure
    assert "total_bridge_nodes" in summary
    assert "persistent_bridge_nodes" in summary
    assert "high_influence_bridge_nodes" in summary
    assert "cross_domain_bridge_nodes" in summary
    assert "bridge_nodes" in summary
    assert "domain_analysis" in summary
    assert "recommendations" in summary
    
    # Check that bridge nodes were tracked
    assert len(integration_with_graph.manager.bridge_history) > 0
    
    # Test with empty graph
    integration = BridgeNodeIntegration()
    summary = await integration.update_bridge_nodes()
    assert summary["total_bridge_nodes"] == 0


async def test_get_bridge_node_recommendations(integration_with_graph):
    """Test getting bridge node recommendations."""
    recommendations = await integration_with_graph.get_bridge_node_recommendations()
    
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
    integration = BridgeNodeIntegration()
    recommendations = await integration.get_bridge_node_recommendations()
    assert len(recommendations) == 0


async def test_get_bridge_node_evolution(integration_with_graph):
    """Test getting bridge node evolution."""
    # Track bridge nodes
    await integration_with_graph.update_bridge_nodes()
    
    # Get a bridge node
    bridge_nodes = await integration_with_graph.manager.identify_bridge_nodes()
    if not bridge_nodes:
        pytest.skip("No bridge nodes found")
    
    node_id = bridge_nodes[0]
    
    analysis = await integration_with_graph.get_bridge_node_evolution(node_id)
    
    # Check analysis structure
    assert "node_id" in analysis
    assert "persistence" in analysis
    assert "stability" in analysis
    assert "influence_trend" in analysis
    assert "community_connections_trend" in analysis
    
    # Test with non-existent node
    analysis = await integration_with_graph.get_bridge_node_evolution("non_existent")
    assert analysis["persistence"] == 0
    assert analysis["influence_trend"] == "unknown"


async def test_get_domain_bridge_analysis(integration_with_graph_and_domains):
    """Test getting domain bridge analysis."""
    analysis = await integration_with_graph_and_domains.get_domain_bridge_analysis()
    
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
    integration = BridgeNodeIntegration(integration_with_graph_and_domains.manager.graph)
    analysis = await integration.get_domain_bridge_analysis()
    assert len(analysis["domains"]) == 0


async def test_get_persistent_bridge_nodes(integration_with_graph):
    """Test getting persistent bridge nodes."""
    # Track multiple times to build history
    for _ in range(5):
        await integration_with_graph.update_bridge_nodes()
    
    persistent_nodes = await integration_with_graph.get_persistent_bridge_nodes()
    
    # Should have some persistent nodes
    assert len(persistent_nodes) > 0
    
    # Test with empty history
    integration = BridgeNodeIntegration()
    persistent_nodes = await integration.get_persistent_bridge_nodes()
    assert len(persistent_nodes) == 0


async def test_get_high_influence_bridge_nodes(integration_with_graph):
    """Test getting high influence bridge nodes."""
    # Track to build history
    await integration_with_graph.update_bridge_nodes()
    
    # Temporarily lower threshold to ensure we get some high influence nodes
    original_threshold = integration_with_graph.manager.influence_threshold
    integration_with_graph.manager.influence_threshold = 0.1
    
    high_influence_nodes = await integration_with_graph.get_high_influence_bridge_nodes()
    
    # Should have some high influence nodes
    assert len(high_influence_nodes) > 0
    
    # Restore threshold
    integration_with_graph.manager.influence_threshold = original_threshold
    
    # Test with empty history
    integration = BridgeNodeIntegration()
    high_influence_nodes = await integration.get_high_influence_bridge_nodes()
    assert len(high_influence_nodes) == 0


async def test_get_cross_domain_bridge_nodes(integration_with_graph_and_domains):
    """Test getting cross-domain bridge nodes."""
    # Track to build history
    await integration_with_graph_and_domains.update_bridge_nodes()
    
    cross_domain_nodes = await integration_with_graph_and_domains.get_cross_domain_bridge_nodes()
    
    # Should have some cross-domain nodes
    assert len(cross_domain_nodes) > 0
    
    # Test with no domain mapping
    integration = BridgeNodeIntegration(integration_with_graph_and_domains.manager.graph)
    await integration.update_bridge_nodes()
    cross_domain_nodes = await integration.get_cross_domain_bridge_nodes()
    assert len(cross_domain_nodes) == 0


async def test_get_bridge_metrics(integration_with_graph_and_domains):
    """Test getting bridge metrics."""
    metrics = await integration_with_graph_and_domains.get_bridge_metrics()
    
    # Check metrics structure
    assert "timestamp" in metrics
    assert "graph_metrics" in metrics
    assert "bridge_summary" in metrics
    assert "domain_analysis" in metrics
    assert "recommendations" in metrics
    
    # Test with empty graph
    integration = BridgeNodeIntegration()
    metrics = await integration.get_bridge_metrics()
    assert "timestamp" in metrics
    assert len(metrics["graph_metrics"]) == 0
    assert len(metrics["bridge_summary"]) == 0


async def test_error_handling():
    """Test error handling in methods."""
    integration = BridgeNodeIntegration()
    
    # Create a graph that will cause errors
    G = MagicMock()
    G.nodes.side_effect = Exception("Test exception")
    integration.set_graph(G)
    
    # Test error handling in various methods
    summary = await integration.update_bridge_nodes()
    assert summary["total_bridge_nodes"] == 0
    
    recommendations = await integration.get_bridge_node_recommendations()
    assert len(recommendations) == 0
    
    analysis = await integration.get_bridge_node_evolution("test")
    assert analysis["persistence"] == 0
    
    domain_analysis = await integration.get_domain_bridge_analysis()
    assert len(domain_analysis["domains"]) == 0
    
    persistent_nodes = await integration.get_persistent_bridge_nodes()
    assert len(persistent_nodes) == 0
    
    high_influence_nodes = await integration.get_high_influence_bridge_nodes()
    assert len(high_influence_nodes) == 0
    
    cross_domain_nodes = await integration.get_cross_domain_bridge_nodes()
    assert len(cross_domain_nodes) == 0
    
    metrics = await integration.get_bridge_metrics()
    assert "timestamp" in metrics
    assert len(metrics["graph_metrics"]) == 0
