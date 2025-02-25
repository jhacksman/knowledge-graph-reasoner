"""Tests for community preservation mechanisms."""
import unittest
import asyncio
import networkx as nx

from unittest.mock import MagicMock, AsyncMock, patch

from src.graph.community_preservation import CommunityPreservation


class TestCommunityPreservation(unittest.TestCase):
    """Test community preservation mechanisms."""

    def setUp(self):
        """Set up test environment."""
        # Create mock graph
        self.graph = nx.Graph()

        # Create community preservation instance
        self.community_preservation = CommunityPreservation(self.graph)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.community_preservation.graph, self.graph)
        self.assertEqual(self.community_preservation.communities, [])
        self.assertEqual(self.community_preservation.node_to_community, {})

    def test_detect_communities_empty(self):
        """Test community detection with empty graph."""
        # Run in event loop
        result = asyncio.run(self.community_preservation.detect_communities())
        self.assertEqual(result, [])

    def test_detect_communities(self):
        """Test community detection with Louvain method."""
        # Setup test graph with clear community structure
        self.graph.add_nodes_from(range(6))
        self.graph.add_edges_from([
            (0, 1), (1, 2),  # Community 1
            (3, 4), (4, 5)   # Community 2
        ])

        # Mock community detection
        with patch('networkx.community.greedy_modularity_communities') as mock_communities:
            mock_communities.return_value = [{0, 1, 2}, {3, 4, 5}]

            # Test community detection
            result = asyncio.run(self.community_preservation.detect_communities())

            # Check results
            self.assertEqual(len(result), 2)
            self.assertEqual(len(self.community_preservation.communities), 2)
            self.assertEqual(self.community_preservation.node_to_community[0], 0)
            self.assertEqual(self.community_preservation.node_to_community[3], 1)

    def test_get_community_metrics_empty(self):
        """Test community metrics with empty graph."""
        # Run in event loop
        result = asyncio.run(self.community_preservation.get_community_metrics())
        self.assertEqual(result["community_count"], 0)

    def test_get_community_metrics(self):
        """Test community metrics with communities."""
        # Setup test graph with communities
        self.graph.add_nodes_from(range(6))
        self.graph.add_edges_from([
            (0, 1), (1, 2),  # Community 1
            (3, 4), (4, 5)   # Community 2
        ])

        # Mock community detection and modularity
        with patch('networkx.community.greedy_modularity_communities') as mock_communities, \
             patch('networkx.community.modularity') as mock_modularity, \
             patch('networkx.density') as mock_density, \
             patch('networkx.average_clustering') as mock_clustering, \
             patch('networkx.diameter') as mock_diameter, \
             patch('networkx.is_connected') as mock_connected:

            mock_communities.return_value = [{0, 1, 2}, {3, 4, 5}]
            mock_modularity.return_value = 0.69
            mock_density.return_value = 0.8
            mock_clustering.return_value = 0.5
            mock_diameter.return_value = 2
            mock_connected.return_value = True

            # Test community metrics
            result = asyncio.run(self.community_preservation.get_community_metrics())

            # Check results
            self.assertEqual(result["community_count"], 2)
            self.assertEqual(result["modularity"], 0.69)
            self.assertEqual(len(result["communities"]), 2)
            self.assertEqual(result["communities"][0]["size"], 3)
            self.assertEqual(result["communities"][0]["density"], 0.8)
            self.assertEqual(result["communities"][0]["avg_clustering"], 0.5)
            self.assertEqual(result["communities"][0]["diameter"], 2)

    def test_preserve_community_structure(self):
        """Test community structure preservation."""
        # Setup test graph with communities
        self.graph.add_nodes_from(range(6))
        self.graph.add_edges_from([
            (0, 1), (1, 2), (0, 2),  # Community 1
            (3, 4), (4, 5), (3, 5)   # Community 2
        ])

        # Mock graph manager and community detection
        graph_manager = MagicMock()
        graph_manager.get_relationships = AsyncMock(return_value=[
            MagicMock(id="rel1", metadata={})
        ])
        graph_manager.update_relationship = AsyncMock()

        with patch('networkx.community.greedy_modularity_communities') as mock_communities, \
             patch('networkx.community.modularity') as mock_modularity, \
             patch.object(CommunityPreservation, 'get_community_metrics') as mock_metrics:

            mock_communities.return_value = [{0, 1, 2}, {3, 4, 5}]
            mock_modularity.return_value = 0.69
            mock_metrics.return_value = {
                "community_count": 2,
                "modularity": 0.69,
                "communities": [
                    {"id": 0, "size": 3},
                    {"id": 1, "size": 3}
                ]
            }

            # Test community preservation
            result = asyncio.run(self.community_preservation.preserve_community_structure(
                graph_manager,
                preservation_factor=0.7
            ))

            # Check results
            self.assertEqual(result["preserved_communities"], 2)
            self.assertTrue(result["strengthened_connections"] > 0)
            self.assertEqual(result["community_metrics"]["community_count"], 2)

    def test_analyze_community_evolution_empty(self):
        """Test community evolution analysis with empty previous communities."""
        # Run in event loop
        result = asyncio.run(self.community_preservation.analyze_community_evolution([]))
        self.assertEqual(result["previous_count"], 0)

    def test_analyze_community_evolution(self):
        """Test community evolution analysis."""
        # Setup test graph with communities
        self.graph.add_nodes_from(range(6))
        self.graph.add_edges_from([
            (0, 1), (1, 2),  # Community 1
            (3, 4), (4, 5)   # Community 2
        ])

        # Mock community detection
        with patch('networkx.community.greedy_modularity_communities') as mock_communities:
            mock_communities.return_value = [{0, 1, 2}, {3, 4, 5}]

            # Detect communities
            asyncio.run(self.community_preservation.detect_communities())

            # Previous communities with slight changes
            previous_communities = [{0, 1}, {2, 3, 4, 5}]

            # Test community evolution
            result = asyncio.run(self.community_preservation.analyze_community_evolution(
                previous_communities
            ))

            # Check results
            self.assertEqual(result["previous_count"], 2)
            self.assertEqual(result["current_count"], 2)
            self.assertEqual(len(result["community_changes"]), 2)
            self.assertTrue(0 <= result["stability_score"] <= 1)

    def test_optimize_modularity(self):
        """Test modularity optimization."""
        # Setup test graph with communities
        self.graph.add_nodes_from(range(6))
        self.graph.add_edges_from([
            (0, 1), (1, 2),  # Community 1
            (3, 4), (4, 5)   # Community 2
        ])

        # Mock graph manager and community detection
        graph_manager = MagicMock()
        graph_manager.get_concept = AsyncMock(return_value=MagicMock(content="Test concept"))
        graph_manager.add_relationship = AsyncMock()

        with patch('networkx.community.greedy_modularity_communities') as mock_communities, \
             patch('networkx.community.modularity') as mock_modularity, \
             patch.object(CommunityPreservation, 'detect_communities') as mock_detect:

            mock_communities.return_value = [{0, 1, 2}, {3, 4, 5}]

            # First call returns low modularity, second call returns target modularity
            mock_modularity.side_effect = [0.5, 0.69]

            # Mock detect_communities to update self.communities
            async def mock_detect_impl():
                self.community_preservation.communities = [set([0, 1, 2]), set([3, 4, 5])]
                return self.community_preservation.communities

            mock_detect.side_effect = mock_detect_impl

            # Test modularity optimization
            result = asyncio.run(self.community_preservation.optimize_modularity(
                graph_manager,
                target_modularity=0.69
            ))

            # Check results
            self.assertEqual(result["initial_modularity"], 0.5)
            self.assertEqual(result["target_modularity"], 0.69)
            self.assertEqual(result["final_modularity"], 0.69)

    def test_error_handling(self):
        """Test error handling."""
        # Mock graph manager with error
        graph_manager = MagicMock()
        graph_manager.get_concept = AsyncMock(side_effect=Exception("Test error"))

        # Test error handling in preserve_community_structure
        result = asyncio.run(self.community_preservation.preserve_community_structure(graph_manager))
        self.assertIn("error", result)
        self.assertEqual(result["preserved_communities"], 0)

        # Test error handling in detect_communities
        with patch('networkx.community.greedy_modularity_communities', side_effect=Exception("Test error")):
            result = asyncio.run(self.community_preservation.detect_communities())
            self.assertEqual(result, [])

        # Test error handling in get_community_metrics
        with patch.object(CommunityPreservation, 'detect_communities', side_effect=Exception("Test error")):
            result = asyncio.run(self.community_preservation.get_community_metrics())
            self.assertIn("error", result)

        # Test error handling in optimize_modularity
        result = asyncio.run(self.community_preservation.optimize_modularity(graph_manager))
        self.assertIn("error", result)
        self.assertEqual(result["initial_modularity"], 0)
        self.assertEqual(result["added_connections"], 0)


if __name__ == "__main__":
    unittest.main()
