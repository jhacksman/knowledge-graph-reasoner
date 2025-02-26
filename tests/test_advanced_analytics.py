"""Tests for advanced analytics module."""
import pytest
import networkx as nx
import numpy as np
from unittest.mock import patch, MagicMock

from src.metrics.advanced_analytics import AdvancedAnalytics


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(range(20))
    
    # Add edges to create a scale-free-like structure
    # Hub nodes: 0, 1, 2
    for i in range(3, 20):
        # Connect to hub nodes with higher probability
        if i % 3 == 0:
            G.add_edge(0, i)
        if i % 4 == 0:
            G.add_edge(1, i)
        if i % 5 == 0:
            G.add_edge(2, i)
        
        # Add some random connections
        G.add_edge(i, (i + 7) % 20)
    
    # Add connections between hubs
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    
    return G


@pytest.fixture
def analytics():
    """Create an AdvancedAnalytics instance."""
    return AdvancedAnalytics()


@pytest.fixture
def analytics_with_graph(sample_graph):
    """Create an AdvancedAnalytics instance with a sample graph."""
    analytics = AdvancedAnalytics(sample_graph)
    return analytics


async def test_init():
    """Test initialization."""
    analytics = AdvancedAnalytics()
    assert isinstance(analytics.graph, nx.Graph)
    assert len(analytics.history) == 0
    assert len(analytics.temporal_snapshots) == 0
    
    # Test with graph
    G = nx.Graph()
    G.add_node(1)
    analytics = AdvancedAnalytics(G)
    assert analytics.graph == G


async def test_set_graph():
    """Test setting graph."""
    analytics = AdvancedAnalytics()
    G = nx.Graph()
    G.add_node(1)
    
    analytics.set_graph(G)
    assert analytics.graph == G


async def test_compute_degree_assortativity(analytics_with_graph):
    """Test computing degree assortativity."""
    assortativity = await analytics_with_graph.compute_degree_assortativity()
    assert isinstance(assortativity, float)
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    assortativity = await analytics.compute_degree_assortativity()
    assert assortativity == 0.0
    
    # Test with single node
    G = nx.Graph()
    G.add_node(1)
    analytics.set_graph(G)
    assortativity = await analytics.compute_degree_assortativity()
    assert assortativity == 0.0


async def test_compute_global_transitivity(analytics_with_graph):
    """Test computing global transitivity."""
    transitivity = await analytics_with_graph.compute_global_transitivity()
    assert isinstance(transitivity, float)
    assert 0.0 <= transitivity <= 1.0
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    transitivity = await analytics.compute_global_transitivity()
    assert transitivity == 0.0


async def test_compute_power_law_exponent(analytics_with_graph):
    """Test computing power law exponent."""
    exponent, fit = await analytics_with_graph.compute_power_law_exponent()
    assert isinstance(exponent, float)
    assert isinstance(fit, float)
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    exponent, fit = await analytics.compute_power_law_exponent()
    assert exponent == 0.0
    assert fit == 0.0


async def test_compute_community_metrics(analytics_with_graph):
    """Test computing community metrics."""
    metrics = await analytics_with_graph.compute_community_metrics()
    assert isinstance(metrics, dict)
    assert "modularity" in metrics
    assert "num_communities" in metrics
    assert "community_sizes" in metrics
    assert "community_density" in metrics
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    metrics = await analytics.compute_community_metrics()
    assert metrics["modularity"] == 0.0
    assert metrics["num_communities"] == 0


async def test_compute_path_metrics(analytics_with_graph):
    """Test computing path metrics."""
    metrics = await analytics_with_graph.compute_path_metrics()
    assert isinstance(metrics, dict)
    assert "avg_path_length" in metrics
    assert "diameter" in metrics
    assert "path_length_distribution" in metrics
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    metrics = await analytics.compute_path_metrics()
    assert metrics["avg_path_length"] == 0.0
    assert metrics["diameter"] == 0.0


async def test_compute_hub_metrics(analytics_with_graph):
    """Test computing hub metrics."""
    metrics = await analytics_with_graph.compute_hub_metrics()
    assert isinstance(metrics, dict)
    assert "hub_nodes" in metrics
    assert "hub_centrality" in metrics
    assert "hub_ratio" in metrics
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    metrics = await analytics.compute_hub_metrics()
    assert len(metrics["hub_nodes"]) == 0
    assert metrics["hub_ratio"] == 0.0


async def test_compute_all_metrics(analytics_with_graph):
    """Test computing all metrics."""
    metrics = await analytics_with_graph.compute_all_metrics()
    assert isinstance(metrics, dict)
    assert "timestamp" in metrics
    assert "graph_size" in metrics
    assert "edge_count" in metrics
    assert "assortativity" in metrics
    assert "transitivity" in metrics
    assert "power_law_exponent" in metrics
    assert "modularity" in metrics
    assert "avg_path_length" in metrics
    assert "diameter" in metrics
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    metrics = await analytics.compute_all_metrics()
    assert metrics["graph_size"] == 0
    assert metrics["edge_count"] == 0


async def test_track_metrics(analytics_with_graph):
    """Test tracking metrics."""
    # Track without snapshot
    metrics = await analytics_with_graph.track_metrics(snapshot_graph=False)
    assert len(analytics_with_graph.history) == 1
    assert len(analytics_with_graph.temporal_snapshots) == 0
    
    # Track with snapshot
    metrics = await analytics_with_graph.track_metrics(snapshot_graph=True)
    assert len(analytics_with_graph.history) == 2
    assert len(analytics_with_graph.temporal_snapshots) == 1
    assert "timestamp" in analytics_with_graph.temporal_snapshots[0]
    assert "graph" in analytics_with_graph.temporal_snapshots[0]
    assert "metrics" in analytics_with_graph.temporal_snapshots[0]


async def test_check_convergence(analytics_with_graph):
    """Test checking convergence."""
    # No history yet
    convergence = await analytics_with_graph.check_convergence()
    assert all(not value for value in convergence.values())
    
    # Add some history
    for _ in range(5):
        await analytics_with_graph.track_metrics()
    
    convergence = await analytics_with_graph.check_convergence()
    assert isinstance(convergence, dict)
    assert "assortativity" in convergence
    assert "transitivity" in convergence
    assert "modularity" in convergence
    assert "avg_path_length" in convergence
    assert "power_law_exponent" in convergence


async def test_analyze_temporal_evolution(analytics_with_graph):
    """Test analyzing temporal evolution."""
    # No history yet
    analysis = await analytics_with_graph.analyze_temporal_evolution("modularity")
    assert analysis["trend"] == "unknown"
    
    # Add some history
    for _ in range(3):
        await analytics_with_graph.track_metrics()
    
    analysis = await analytics_with_graph.analyze_temporal_evolution("modularity")
    assert "trend" in analysis
    assert "stability" in analysis
    assert "rate_of_change" in analysis


async def test_validate_scale_free_property(analytics_with_graph):
    """Test validating scale-free property."""
    validation = await analytics_with_graph.validate_scale_free_property()
    assert isinstance(validation, dict)
    assert "is_scale_free" in validation
    assert "power_law_exponent" in validation
    assert "power_law_fit" in validation
    assert "degree_distribution" in validation
    
    # Test with empty graph
    analytics = AdvancedAnalytics()
    validation = await analytics.validate_scale_free_property()
    assert not validation["is_scale_free"]
    assert validation["power_law_exponent"] == 0.0


async def test_get_metric_history(analytics_with_graph):
    """Test getting metric history."""
    # No history yet
    history = await analytics_with_graph.get_metric_history("modularity")
    assert len(history) == 0
    
    # Add some history
    for _ in range(3):
        await analytics_with_graph.track_metrics()
    
    history = await analytics_with_graph.get_metric_history("modularity")
    assert len(history) == 3
    assert all(isinstance(value, float) for value in history)


async def test_get_summary_report(analytics_with_graph):
    """Test getting summary report."""
    # No history yet
    report = await analytics_with_graph.get_summary_report()
    assert report["status"] == "No data available"
    
    # Add some history
    for _ in range(3):
        await analytics_with_graph.track_metrics()
    
    report = await analytics_with_graph.get_summary_report()
    assert "status" in report
    assert "metrics" in report
    assert "convergence" in report
    assert "recommendations" in report
    assert isinstance(report["recommendations"], list)


async def test_error_handling():
    """Test error handling in methods."""
    analytics = AdvancedAnalytics()
    
    # Create a graph that will cause errors
    G = MagicMock()
    G.nodes.side_effect = Exception("Test exception")
    analytics.set_graph(G)
    
    # Test error handling in various methods
    assortativity = await analytics.compute_degree_assortativity()
    assert assortativity == 0.0
    
    transitivity = await analytics.compute_global_transitivity()
    assert transitivity == 0.0
    
    exponent, fit = await analytics.compute_power_law_exponent()
    assert exponent == 0.0
    assert fit == 0.0
    
    metrics = await analytics.compute_community_metrics()
    assert metrics["modularity"] == 0.0
    
    metrics = await analytics.compute_path_metrics()
    assert metrics["avg_path_length"] == 0.0
    
    metrics = await analytics.compute_hub_metrics()
    assert len(metrics["hub_nodes"]) == 0
