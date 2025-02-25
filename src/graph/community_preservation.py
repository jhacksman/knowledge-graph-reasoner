"""Community preservation mechanisms for self-organizing knowledge networks."""
from typing import List, Dict, Any, Set
import networkx as nx
import logging
import numpy as np

log = logging.getLogger(__name__)


class CommunityPreservation:
    """Implements community preservation mechanisms for knowledge graphs."""

    def __init__(self, graph: nx.Graph):
        """Initialize community preservation.

        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self.communities: List[Set[str]] = []
        self.node_to_community: Dict[str, int] = {}

    async def detect_communities(self) -> List[Set[str]]:
        """Detect communities using Louvain method.

        Returns:
            List[Set[str]]: List of communities (sets of node IDs)
        """
        try:
            # Detect communities using Louvain method
            communities = nx.community.greedy_modularity_communities(self.graph)

            # Convert to list of sets
            self.communities = [set(c) for c in communities]

            # Create node to community mapping
            self.node_to_community = {}
            for i, community in enumerate(self.communities):
                for node in community:
                    self.node_to_community[node] = i

            return self.communities

        except Exception as e:
            log.error(f"Failed to detect communities: {e}")
            return []

    async def get_community_metrics(self) -> Dict[str, Any]:
        """Get metrics for each community.

        Returns:
            Dict[str, Any]: Community metrics
        """
        try:
            if not self.communities:
                await self.detect_communities()

            metrics = {
                "community_count": len(self.communities),
                "community_sizes": [len(c) for c in self.communities],
                "modularity": nx.community.modularity(self.graph, self.communities),
                "communities": []
            }

            # Calculate metrics for each community
            for i, community in enumerate(self.communities):
                subgraph = self.graph.subgraph(community)

                # Calculate community metrics
                community_metrics = {
                    "id": i,
                    "size": len(community),
                    "density": nx.density(subgraph),
                    "avg_clustering": nx.average_clustering(subgraph),
                    "diameter": nx.diameter(subgraph) if nx.is_connected(subgraph) else -1
                }

                metrics["communities"].append(community_metrics)

            return metrics

        except Exception as e:
            log.error(f"Failed to get community metrics: {e}")
            return {
                "error": str(e),
                "community_count": len(self.communities) if self.communities else 0
            }

    async def preserve_community_structure(
        self,
        graph_manager,
        preservation_factor: float = 0.7
    ) -> Dict[str, Any]:
        """Preserve community structure during graph expansion.

        Args:
            graph_manager: Graph manager instance
            preservation_factor: Factor for community preservation (0-1)

        Returns:
            Dict[str, Any]: Results of community preservation
        """
        try:
            if not self.communities:
                await self.detect_communities()

            results = {
                "preserved_communities": len(self.communities),
                "strengthened_connections": 0,
                "community_metrics": await self.get_community_metrics()
            }

            # Strengthen intra-community connections
            for community in self.communities:
                # Get all pairs of nodes in the community
                for node1 in community:
                    for node2 in community:
                        if node1 != node2:
                            # Check if already connected
                            if self.graph.has_edge(node1, node2):
                                # Strengthen existing connection
                                relationships = await graph_manager.get_relationships(
                                    source_id=node1,
                                    target_id=node2
                                )

                                for rel in relationships:
                                    # Update relationship metadata
                                    metadata = rel.metadata or {}
                                    metadata["community_preserved"] = True
                                    metadata["community_id"] = self.node_to_community.get(node1)

                                    # Update relationship
                                    await graph_manager.update_relationship(
                                        rel.id,
                                        metadata=metadata
                                    )

                                    results["strengthened_connections"] += 1

            return results

        except Exception as e:
            log.error(f"Failed to preserve community structure: {e}")
            return {
                "error": str(e),
                "preserved_communities": 0,
                "strengthened_connections": 0
            }

    async def analyze_community_evolution(
        self,
        previous_communities: List[Set[str]]
    ) -> Dict[str, Any]:
        """Analyze community evolution between iterations.

        Args:
            previous_communities: Communities from previous iteration

        Returns:
            Dict[str, Any]: Community evolution analysis
        """
        try:
            if not self.communities:
                await self.detect_communities()

            current_communities = self.communities

            # Calculate community stability metrics
            stability_metrics = {
                "previous_count": len(previous_communities),
                "current_count": len(current_communities),
                "community_changes": []
            }

            # Calculate Jaccard similarity between communities
            similarity_matrix = []
            for prev_comm in previous_communities:
                similarities = []
                for curr_comm in current_communities:
                    # Calculate Jaccard similarity
                    intersection = len(prev_comm.intersection(curr_comm))
                    union = len(prev_comm.union(curr_comm))
                    similarity = float(intersection) / float(union) if union > 0 else 0.0
                    similarities.append(similarity)
                similarity_matrix.append(similarities)

            # Find best matches between previous and current communities
            matched_communities = []
            for i, prev_similarities in enumerate(similarity_matrix):
                if prev_similarities:  # Check if not empty
                    best_match = np.argmax(prev_similarities)
                    best_similarity = prev_similarities[best_match]

                    matched_communities.append({
                        "previous_id": i,
                        "current_id": best_match,
                        "similarity": best_similarity,
                        "previous_size": len(previous_communities[i]),
                        "current_size": len(current_communities[best_match]) if best_match < len(current_communities) else 0
                    })

            stability_metrics["community_changes"] = matched_communities

            # Calculate overall stability score (average of best match similarities)
            if matched_communities:
                stability_metrics["stability_score"] = sum(float(m["similarity"]) for m in matched_communities) / len(matched_communities)
            else:
                stability_metrics["stability_score"] = 0.0

            return stability_metrics

        except Exception as e:
            log.error(f"Failed to analyze community evolution: {e}")
            return {
                "error": str(e),
                "previous_count": len(previous_communities),
                "current_count": len(self.communities) if self.communities else 0,
                "stability_score": 0
            }

    async def optimize_modularity(
        self,
        graph_manager,
        target_modularity: float = 0.69
    ) -> Dict[str, Any]:
        """Optimize graph structure to achieve target modularity.

        Args:
            graph_manager: Graph manager instance
            target_modularity: Target modularity score

        Returns:
            Dict[str, Any]: Results of modularity optimization
        """
        try:
            if not self.communities:
                await self.detect_communities()

            # Get current modularity
            current_modularity = nx.community.modularity(self.graph, self.communities)

            results = {
                "initial_modularity": current_modularity,
                "target_modularity": target_modularity,
                "added_connections": 0,
                "final_modularity": current_modularity
            }

            # If current modularity is already close to target, no need to optimize
            if abs(current_modularity - target_modularity) < 0.05:
                return results

            # Identify communities that need strengthening
            for i, community in enumerate(self.communities):
                # For small communities, strengthen internal connections
                if len(community) >= 3:
                    # Find node pairs that are not connected
                    for node1 in community:
                        for node2 in community:
                            if node1 != node2 and not self.graph.has_edge(node1, node2):
                                # Add new connection to strengthen community
                                try:
                                    # Get node content
                                    node1_obj = await graph_manager.get_concept(node1)
                                    node2_obj = await graph_manager.get_concept(node2)

                                    if node1_obj and node2_obj:
                                        # Create relationship
                                        await graph_manager.add_relationship(
                                            node1,
                                            node2,
                                            "community_optimized",
                                            {
                                                "description": f"Connection to optimize modularity between: {node1_obj.content} and {node2_obj.content}",
                                                "community_id": i,
                                                "modularity_optimization": True
                                            }
                                        )

                                        results["added_connections"] += 1
                                except Exception as e:
                                    log.error(f"Failed to add relationship for modularity optimization: {e}")

            # Recalculate communities and modularity
            if results["added_connections"] > 0:
                await self.detect_communities()
                results["final_modularity"] = nx.community.modularity(self.graph, self.communities)

            return results

        except Exception as e:
            log.error(f"Failed to optimize modularity: {e}")
            return {
                "error": str(e),
                "initial_modularity": 0,
                "target_modularity": target_modularity,
                "added_connections": 0,
                "final_modularity": 0
            }
