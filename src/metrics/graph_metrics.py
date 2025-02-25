"""Graph metrics implementation."""
from typing import List, Dict, Any
import numpy as np
import networkx as nx
import logging

log = logging.getLogger(__name__)


class GraphMetrics:
    """Computes and tracks graph metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.graph = nx.Graph()
    
    async def compute_modularity(self) -> float:
        """Compute graph modularity score.
        
        Returns:
            float: Modularity score (0-1)
        """
        try:
            if len(self.graph) < 2:
                return 0.0
            
            communities = nx.community.greedy_modularity_communities(self.graph)
            return nx.community.modularity(self.graph, communities)
        except Exception as e:
            log.error(f"Failed to compute modularity: {e}")
            return 0.0
    
    async def compute_avg_path_length(self) -> float:
        """Compute average shortest path length.
        
        Returns:
            float: Average path length
        """
        try:
            if len(self.graph) < 2:
                return 0.0
            
            # Only consider largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            
            return nx.average_shortest_path_length(subgraph)
        except Exception as e:
            log.error(f"Failed to compute average path length: {e}")
            return 0.0
    
    async def compute_diameter(self) -> float:
        """Compute graph diameter.
        
        Returns:
            float: Graph diameter
        """
        try:
            if len(self.graph) < 2:
                return 0.0
            
            # Only consider largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            
            return nx.diameter(subgraph)
        except Exception as e:
            log.error(f"Failed to compute diameter: {e}")
            return 0.0
    
    async def find_bridge_nodes(self) -> List[str]:
        """Identify bridge nodes connecting communities.
        
        Returns:
            List[str]: IDs of bridge nodes
        """
        try:
            if len(self.graph) < 3:
                return []
            
            # Find nodes with high betweenness centrality
            centrality = nx.betweenness_centrality(self.graph)
            
            # Consider top 10% of nodes as bridges
            threshold = np.percentile(
                list(centrality.values()),
                90
            )
            
            return [
                node_id for node_id, score in centrality.items()
                if score >= threshold
            ]
        except Exception as e:
            log.error(f"Failed to find bridge nodes: {e}")
            return []
    
    def update_graph(self, nodes: List[str], edges: List[Dict[str, Any]]) -> None:
        """Update graph structure.
        
        Args:
            nodes: List of node IDs
            edges: List of edges as dicts with source, target keys
        """
        try:
            # Add nodes
            self.graph.add_nodes_from(nodes)
            
            # Add edges
            edge_tuples = [
                (edge["source"], edge["target"])
                for edge in edges
            ]
            self.graph.add_edges_from(edge_tuples)
        except Exception as e:
            log.error(f"Failed to update graph: {e}")
