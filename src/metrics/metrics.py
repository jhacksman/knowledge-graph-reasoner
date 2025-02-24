"""Graph metrics implementation."""
from typing import List, Dict, Any, Optional, Set
import numpy as np
import networkx as nx
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


class GraphMetrics:
    """Computes and tracks graph metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.graph = nx.Graph()
        self.history: List[Dict[str, Any]] = []
    
    async def compute_modularity(self) -> float:
        """Compute graph modularity score.
        
        Returns:
            float: Modularity score (0-1)
        """
        try:
            if len(self.graph) < 2:
                return 0.0
            
            # Find communities using Louvain method
            communities = nx.community.louvain_communities(self.graph)
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
            
            return float(nx.diameter(subgraph))
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
            
            # Find communities
            communities = nx.community.louvain_communities(self.graph)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Find nodes connecting different communities
            bridge_nodes = set()
            for node in self.graph.nodes():
                node_community = community_map[node]
                neighbor_communities = {
                    community_map[neighbor]
                    for neighbor in self.graph.neighbors(node)
                }
                
                # Node is a bridge if it connects to multiple communities
                if len(neighbor_communities) > 1:
                    bridge_nodes.add(node)
            
            # Sort by betweenness centrality
            centrality = nx.betweenness_centrality(self.graph)
            return sorted(
                bridge_nodes,
                key=lambda x: centrality[x],
                reverse=True
            )
        except Exception as e:
            log.error(f"Failed to find bridge nodes: {e}")
            return []
    
    async def compute_hub_centrality(self) -> Dict[str, float]:
        """Compute hub centrality scores.
        
        Returns:
            Dict[str, float]: Node ID to centrality score mapping
        """
        try:
            if len(self.graph) == 0:
                return {}
            elif len(self.graph) == 1:
                # Single node has maximum centrality
                return {list(self.graph.nodes())[0]: 1.0}
            
            # Use eigenvector centrality as hub measure
            return nx.eigenvector_centrality(self.graph)
        except Exception as e:
            log.error(f"Failed to compute hub centrality: {e}")
            return {}
    
    async def compute_path_length_distribution(self) -> Dict[int, int]:
        """Compute path length distribution.
        
        Returns:
            Dict[int, int]: Path length to count mapping
        """
        try:
            if len(self.graph) < 2:
                return {}
            
            # Only consider largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            
            # Compute all shortest paths
            distribution: Dict[int, int] = defaultdict(int)
            for source in subgraph:
                path_lengths = nx.single_source_shortest_path_length(
                    subgraph,
                    source
                )
                for length in path_lengths.values():
                    distribution[length] += 1
            
            return dict(distribution)
        except Exception as e:
            log.error(f"Failed to compute path distribution: {e}")
            return {}
    
    async def check_stability(
        self,
        window_size: int = 5,
        min_path_length: float = 4.5,
        max_path_length: float = 5.0,
        min_diameter: float = 16.0,
        max_diameter: float = 18.0
    ) -> bool:
        """Check if graph has reached stable state.
        
        Args:
            window_size: Number of historical states to consider
            min_path_length: Target minimum average path length
            max_path_length: Target maximum average path length
            min_diameter: Target minimum graph diameter
            max_diameter: Target maximum graph diameter
            
        Returns:
            bool: True if stable
        """
        try:
            if len(self.history) < window_size:
                return False
            
            # Get recent metrics
            recent = self.history[-window_size:]
            
            # Check path length stability
            path_lengths = [m["avg_path_length"] for m in recent]
            if not all(min_path_length <= l <= max_path_length
                      for l in path_lengths):
                return False
            
            # Check diameter stability
            diameters = [m.get("diameter", 0) for m in recent]
            if not all(min_diameter <= d <= max_diameter
                      for d in diameters):
                return False
            
            # Check modularity trend
            modularity_values = [m["modularity"] for m in recent]
            if len(modularity_values) > 1:
                # Should not be decreasing significantly
                changes = np.diff(modularity_values)
                if any(change < -0.1 for change in changes):
                    return False
            
            return True
        except Exception as e:
            log.error(f"Failed to check stability: {e}")
            return False
    
    async def update_graph(
        self,
        nodes: List[str],
        edges: List[Dict[str, Any]],
        track_metrics: bool = True
    ) -> None:
        """Update graph structure.
        
        Args:
            nodes: List of node IDs
            edges: List of edges as dicts with source, target keys
            track_metrics: Whether to compute and track metrics
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
            
            # Track metrics if requested
            if track_metrics:
                metrics = {
                    "modularity": await self.compute_modularity(),
                    "avg_path_length": await self.compute_avg_path_length(),
                    "diameter": await self.compute_diameter(),
                    "bridge_nodes": await self.find_bridge_nodes()
                }
                self.history.append(metrics)
        except Exception as e:
            log.error(f"Failed to update graph: {e}")
