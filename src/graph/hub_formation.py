"""Hub formation algorithms for self-organizing knowledge networks."""
from typing import List, Dict, Any, Collection
import numpy as np
import networkx as nx
import logging

from ..metrics.graph_metrics import GraphMetrics

log = logging.getLogger(__name__)


class HubFormation:
    """Implements hub formation algorithms for knowledge graphs."""

    def __init__(self, metrics: GraphMetrics):
        """Initialize hub formation.

        Args:
            metrics: Graph metrics instance
        """
        self.metrics = metrics

    async def identify_potential_hubs(self, threshold_percentile: float = 90) -> List[str]:
        """Identify potential hub nodes based on centrality measures.

        Args:
            threshold_percentile: Percentile threshold for hub identification

        Returns:
            List[str]: List of potential hub node IDs
        """
        try:
            # Get centrality measures
            centrality = nx.degree_centrality(self.metrics.graph)
            betweenness = nx.betweenness_centrality(self.metrics.graph)
            eigenvector = nx.eigenvector_centrality(self.metrics.graph, max_iter=1000)

            # Combine centrality measures
            combined_centrality = {}
            for node in self.metrics.graph.nodes():
                combined_centrality[node] = (
                    centrality.get(node, 0) +
                    betweenness.get(node, 0) +
                    eigenvector.get(node, 0)
                ) / 3

            # Identify hubs based on threshold
            threshold = np.percentile(
                list(combined_centrality.values()),
                threshold_percentile
            )

            return [
                str(node_id) for node_id, score in combined_centrality.items()
                if score >= threshold
            ]

        except Exception as e:
            log.error(f"Failed to identify potential hubs: {e}")
            return []

    async def strengthen_hub_connections(
        self,
        hub_ids: List[str],
        graph_manager
    ) -> Dict[str, Any]:
        """Strengthen connections to and between hub nodes.

        Args:
            hub_ids: List of hub node IDs
            graph_manager: Graph manager instance

        Returns:
            Dict[str, Any]: Results of hub strengthening
        """
        try:
            results = {
                "strengthened_connections": 0,
                "new_connections": 0,
                "hub_count": len(hub_ids)
            }

            # Strengthen existing connections
            for hub_id in hub_ids:
                # Get hub node
                hub_node = await graph_manager.get_concept(hub_id)
                if not hub_node:
                    continue

                # Get connected nodes
                connected_nodes = await graph_manager.get_connected_concepts(hub_id)

                # Strengthen existing connections
                for connected_node in connected_nodes:
                    # Update relationship metadata to indicate strengthened connection
                    relationships = await graph_manager.get_relationships(
                        source_id=hub_id,
                        target_id=connected_node.id
                    )

                    for rel in relationships:
                        # Update relationship metadata
                        metadata = rel.metadata or {}
                        metadata["hub_strengthened"] = True
                        metadata["hub_strength"] = int(metadata.get("hub_strength", 0)) + 1

                        # Update relationship
                        await graph_manager.update_relationship(
                            rel.id,
                            metadata=metadata
                        )

                        results["strengthened_connections"] += 1

            # Connect hubs to each other if not already connected
            for i, hub1_id in enumerate(hub_ids):
                for hub2_id in hub_ids[i+1:]:
                    # Check if already connected
                    relationships = await graph_manager.get_relationships(
                        source_id=hub1_id,
                        target_id=hub2_id
                    )

                    if not relationships:
                        # Get hub nodes
                        hub1 = await graph_manager.get_concept(hub1_id)
                        hub2 = await graph_manager.get_concept(hub2_id)

                        if hub1 and hub2:
                            # Create new relationship
                            await graph_manager.add_relationship(
                                hub1_id,
                                hub2_id,
                                "hub_connection",
                                {
                                    "description": f"Connection between hub concepts: {hub1.content} and {hub2.content}",
                                    "hub_connection": True
                                }
                            )

                            results["new_connections"] += 1

            return results

        except Exception as e:
            log.error(f"Failed to strengthen hub connections: {e}")
            return {
                "error": str(e),
                "strengthened_connections": 0,
                "new_connections": 0,
                "hub_count": len(hub_ids)
            }

    async def analyze_hub_structure(self) -> Dict[str, Any]:
        """Analyze the hub structure of the graph.

        Returns:
            Dict[str, Any]: Hub structure analysis
        """
        try:
            # Get centrality measures
            degree = nx.degree_centrality(self.metrics.graph)
            betweenness = nx.betweenness_centrality(self.metrics.graph)
            eigenvector = nx.eigenvector_centrality(self.metrics.graph, max_iter=1000)

            # Calculate hub metrics
            hub_metrics = {
                "degree_distribution": self._calculate_distribution(degree.values()),
                "betweenness_distribution": self._calculate_distribution(betweenness.values()),
                "eigenvector_distribution": self._calculate_distribution(eigenvector.values()),
                "top_hubs": []  # type: ignore
            }

            # Identify top hubs
            combined_centrality = {}
            for node in self.metrics.graph.nodes():
                combined_centrality[node] = (
                    degree.get(node, 0) +
                    betweenness.get(node, 0) +
                    eigenvector.get(node, 0)
                ) / 3

            # Get top 10 hubs
            top_hubs = sorted(
                combined_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            # Convert to list of dictionaries with string IDs
            for hub_id, score in top_hubs:
                hub_metrics["top_hubs"].append({
                    "id": str(hub_id),
                    "centrality": score,
                    "degree": degree.get(hub_id, 0),
                    "betweenness": betweenness.get(hub_id, 0),
                    "eigenvector": eigenvector.get(hub_id, 0)
                }
            # Assign the properly typed list
            hub_metrics["top_hubs"] = top_hubs_list

            # Check for scale-free properties
            hub_metrics["scale_free_properties"] = self._check_scale_free_properties()

            return hub_metrics

        except Exception as e:
            log.error(f"Failed to analyze hub structure: {e}")
            return {
                "error": str(e)
            }

    def _calculate_distribution(self, values) -> Dict[str, float]:
        """Calculate distribution statistics.

        Args:
            values: List of values

        Returns:
            Dict[str, float]: Distribution statistics
        """
        values_list = list(values)
        return {
            "min": min(values_list) if values_list else 0,
            "max": max(values_list) if values_list else 0,
            "mean": np.mean(values_list) if values_list else 0,
            "median": np.median(values_list) if values_list else 0,
            "std": np.std(values_list) if values_list else 0,
            "percentile_90": np.percentile(values_list, 90) if values_list else 0,
            "percentile_95": np.percentile(values_list, 95) if values_list else 0
        }

    def _check_scale_free_properties(self) -> Dict[str, Any]:
        """Check for scale-free properties in the graph.

        Returns:
            Dict[str, Any]: Scale-free properties analysis
        """
        try:
            # Get degree distribution
            degrees = [d for _, d in self.metrics.graph.degree()]

            # Check if empty
            if not degrees:
                return {
                    "is_scale_free": False,
                    "power_law_exponent": None,
                    "r_squared": None
                }

            # Calculate degree distribution
            unique_degrees, counts = np.unique(degrees, return_counts=True)

            # Filter out zeros
            non_zero_indices = unique_degrees > 0
            unique_degrees = unique_degrees[non_zero_indices]
            counts = counts[non_zero_indices]

            # Check if we have enough data points
            if len(unique_degrees) < 5:
                return {
                    "is_scale_free": False,
                    "power_law_exponent": None,
                    "r_squared": None,
                    "reason": "Not enough data points"
                }

            # Log-log transformation
            log_degrees = np.log(unique_degrees)
            log_counts = np.log(counts)

            # Linear regression on log-log data
            coeffs = np.polyfit(log_degrees, log_counts, 1)
            power_law_exponent = -coeffs[0]  # Negative slope

            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_degrees)
            ss_total = np.sum((log_counts - np.mean(log_counts))**2)
            ss_residual = np.sum((log_counts - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)

            # Check if scale-free (power law exponent around 2-3 and good fit)
            is_scale_free = (2.0 <= power_law_exponent <= 3.5) and (r_squared >= 0.8)

            return {
                "is_scale_free": is_scale_free,
                "power_law_exponent": power_law_exponent,
                "r_squared": r_squared,
                "degree_distribution": {
                    "degrees": unique_degrees.tolist(),
                    "counts": counts.tolist()
                }
            }

        except Exception as e:
            log.error(f"Failed to check scale-free properties: {e}")
            return {
                "is_scale_free": False,
                "power_law_exponent": None,
                "r_squared": None,
                "error": str(e)
            }
