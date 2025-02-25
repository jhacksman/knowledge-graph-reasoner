"""Tests for hub formation mechanisms."""
import unittest
import asyncio
import networkx as nx

from unittest.mock import MagicMock, AsyncMock, patch

from src.graph.hub_formation import HubFormation
from src.metrics.graph_metrics import GraphMetrics


class TestHubFormation(unittest.TestCase):
    """Test hub formation mechanisms."""

    def setUp(self):
        """Set up test environment."""
        # Create mock graph metrics
        self.metrics = MagicMock(spec=GraphMetrics)
        self.metrics.graph = nx.Graph()

        # Create hub formation instance
        self.hub_formation = HubFormation(self.metrics)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.hub_formation.metrics, self.metrics)

    def test_calculate_distribution(self):
        """Test distribution calculation."""
        # Test with empty values
        result = self.hub_formation._calculate_distribution([])
        self.assertEqual(result["min"], 0)
        self.assertEqual(result["max"], 0)
        self.assertEqual(result["mean"], 0)

        # Test with values
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = self.hub_formation._calculate_distribution(values)
        self.assertEqual(result["min"], 0.1)
        self.assertEqual(result["max"], 0.5)
        self.assertEqual(result["mean"], 0.3)
        self.assertEqual(result["median"], 0.3)
        self.assertAlmostEqual(result["std"], 0.1414, places=4)

    def test_check_scale_free_properties_empty(self):
        """Test scale-free properties check with empty graph."""
        # Empty graph
        result = self.hub_formation._check_scale_free_properties()
        self.assertFalse(result["is_scale_free"])
        self.assertIsNone(result["power_law_exponent"])
        self.assertIsNone(result["r_squared"])

    def test_check_scale_free_properties_small(self):
        """Test scale-free properties check with small graph."""
        # Small graph (not enough data points)
        self.metrics.graph.add_nodes_from(range(3))
        self.metrics.graph.add_edges_from([(0, 1), (1, 2)])

        result = self.hub_formation._check_scale_free_properties()
        self.assertFalse(result["is_scale_free"])
        self.assertIsNone(result["power_law_exponent"])
        self.assertIsNone(result["r_squared"])

    def test_check_scale_free_properties_scale_free(self):
        """Test scale-free properties check with scale-free graph."""
        # Create a scale-free graph
        G = nx.barabasi_albert_graph(100, 2, seed=42)
        self.metrics.graph = G

        result = self.hub_formation._check_scale_free_properties()

        # Check if power law exponent is in the expected range for scale-free networks
        self.assertIsNotNone(result["power_law_exponent"])
        self.assertIsNotNone(result["r_squared"])

        # Scale-free networks typically have power law exponent between 2 and 3
        if result["is_scale_free"]:
            self.assertTrue(2.0 <= result["power_law_exponent"] <= 3.5)

    def test_identify_potential_hubs_empty(self):
        """Test hub identification with empty graph."""
        # Run in event loop
        result = asyncio.run(self.hub_formation.identify_potential_hubs())
        self.assertEqual(result, [])

    def test_identify_potential_hubs(self):
        """Test hub identification with different thresholds."""
        # Setup test graph
        self.metrics.graph.add_nodes_from(range(10))
        self.metrics.graph.add_edges_from([
            (0, 1), (0, 2), (0, 3), (0, 4),  # Node 0 is a hub
            (1, 2), (2, 3),
            (4, 5), (5, 6),
            (7, 8), (8, 9)
        ])

        # Mock centrality measures
        with patch('networkx.degree_centrality') as mock_degree, \
             patch('networkx.betweenness_centrality') as mock_betweenness, \
             patch('networkx.eigenvector_centrality') as mock_eigenvector:

            # Set up mock return values
            mock_degree.return_value = {
                0: 0.4, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2,
                5: 0.2, 6: 0.1, 7: 0.1, 8: 0.2, 9: 0.1
            }
            mock_betweenness.return_value = {
                0: 0.5, 1: 0.1, 2: 0.3, 3: 0.1, 4: 0.1,
                5: 0.2, 6: 0.0, 7: 0.0, 8: 0.1, 9: 0.0
            }
            mock_eigenvector.return_value = {
                0: 0.6, 1: 0.3, 2: 0.4, 3: 0.2, 4: 0.3,
                5: 0.2, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.0
            }

            # Test with 90th percentile threshold
            result = asyncio.run(self.hub_formation.identify_potential_hubs(threshold_percentile=90))
            self.assertEqual(len(result), 1)
            self.assertIn(0, result)  # Node 0 should be identified as hub

            # Test with lower threshold
            result = asyncio.run(self.hub_formation.identify_potential_hubs(threshold_percentile=70))
            self.assertTrue(len(result) > 1)

    def test_strengthen_hub_connections(self):
        """Test strengthening hub connections."""
        # Setup test graph
        self.metrics.graph.add_nodes_from(range(5))
        self.metrics.graph.add_edges_from([
            (0, 1), (0, 2), (0, 3),  # Node 0 is a hub
            (1, 2), (2, 3), (3, 4)
        ])

        # Mock graph manager
        graph_manager = MagicMock()
        graph_manager.get_concept = AsyncMock(return_value=MagicMock(content="Test concept", id=0))
        graph_manager.get_connected_concepts = AsyncMock(return_value=[
            MagicMock(content="Connected 1", id=1),
            MagicMock(content="Connected 2", id=2),
            MagicMock(content="Connected 3", id=3)
        ])
        graph_manager.get_relationships = AsyncMock(return_value=[
            MagicMock(id="rel1", metadata={})
        ])
        graph_manager.update_relationship = AsyncMock()
        graph_manager.add_relationship = AsyncMock()

        # Test strengthening connections
        result = asyncio.run(self.hub_formation.strengthen_hub_connections([0], graph_manager))

        # Check results
        self.assertEqual(result["hub_count"], 1)
        self.assertEqual(result["strengthened_connections"], 3)  # 3 connections strengthened
        self.assertEqual(result["new_connections"], 0)  # No new connections

        # Test with multiple hubs
        graph_manager.get_relationships.return_value = []  # No existing relationships
        result = asyncio.run(self.hub_formation.strengthen_hub_connections([0, 4], graph_manager))

        # Check results
        self.assertEqual(result["hub_count"], 2)
        self.assertEqual(result["new_connections"], 1)  # New connection between hubs

    def test_analyze_hub_structure(self):
        """Test hub structure analysis."""
        # Setup test graph
        self.metrics.graph.add_nodes_from(range(10))
        self.metrics.graph.add_edges_from([
            (0, 1), (0, 2), (0, 3), (0, 4),  # Node 0 is a hub
            (1, 2), (2, 3),
            (4, 5), (5, 6),
            (7, 8), (8, 9)
        ])

        # Mock centrality measures and scale-free check
        with patch.object(HubFormation, '_check_scale_free_properties') as mock_scale_free, \
             patch('networkx.degree_centrality') as mock_degree, \
             patch('networkx.betweenness_centrality') as mock_betweenness, \
             patch('networkx.eigenvector_centrality') as mock_eigenvector:

            # Set up mock return values
            mock_scale_free.return_value = {
                "is_scale_free": True,
                "power_law_exponent": 2.8,
                "r_squared": 0.92
            }
            mock_degree.return_value = {
                0: 0.4, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2,
                5: 0.2, 6: 0.1, 7: 0.1, 8: 0.2, 9: 0.1
            }
            mock_betweenness.return_value = {
                0: 0.5, 1: 0.1, 2: 0.3, 3: 0.1, 4: 0.1,
                5: 0.2, 6: 0.0, 7: 0.0, 8: 0.1, 9: 0.0
            }
            mock_eigenvector.return_value = {
                0: 0.6, 1: 0.3, 2: 0.4, 3: 0.2, 4: 0.3,
                5: 0.2, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.0
            }

            # Test hub structure analysis
            result = asyncio.run(self.hub_formation.analyze_hub_structure())

            # Check results
            self.assertTrue(result["scale_free_properties"]["is_scale_free"])
            self.assertEqual(result["scale_free_properties"]["power_law_exponent"], 2.8)
            self.assertEqual(len(result["top_hubs"]), 10)
            self.assertEqual(result["top_hubs"][0]["id"], 0)  # Node 0 should be top hub

    def test_error_handling(self):
        """Test error handling."""
        # Mock graph manager with error
        graph_manager = MagicMock()
        graph_manager.get_concept = AsyncMock(side_effect=Exception("Test error"))

        # Test error handling in strengthen_hub_connections
        result = asyncio.run(self.hub_formation.strengthen_hub_connections([0], graph_manager))

        # Check results
        self.assertIn("error", result)
        self.assertEqual(result["strengthened_connections"], 0)
        self.assertEqual(result["new_connections"], 0)

        # Test error handling in identify_potential_hubs
        with patch('networkx.degree_centrality', side_effect=Exception("Test error")):
            result = asyncio.run(self.hub_formation.identify_potential_hubs())
            self.assertEqual(result, [])

        # Test error handling in analyze_hub_structure
        with patch('networkx.degree_centrality', side_effect=Exception("Test error")):
            result = asyncio.run(self.hub_formation.analyze_hub_structure())
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
