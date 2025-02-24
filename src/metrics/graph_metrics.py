"""Graph metrics computation module."""
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
from community import community_louvain

from ..models.node import Node
from ..models.edge import Edge


class GraphMetricsComputer:
    """Computes various metrics for knowledge graph analysis."""
    
    @staticmethod
    def _build_networkx_graph(nodes: List[Node], edges: List[Edge]) -> nx.Graph:
        """Build a NetworkX graph from nodes and edges.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            NetworkX graph instance
        """
        G = nx.Graph()
        
        # Add nodes with attributes
        for node in nodes:
            G.add_node(node.id, content=node.content, metadata=node.metadata)
            
        # Add edges with attributes
        for edge in edges:
            G.add_edge(edge.source, edge.target, type=edge.type, metadata=edge.metadata)
            
        return G
    
    @staticmethod
    def compute_avg_path_length(nodes: List[Node], edges: List[Edge]) -> float:
        """Compute average shortest path length.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            Average shortest path length
        """
        G = GraphMetricsComputer._build_networkx_graph(nodes, edges)
        if not nx.is_connected(G):
            # Handle disconnected components by computing for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
        return nx.average_shortest_path_length(G)
    
    @staticmethod
    def compute_diameter(nodes: List[Node], edges: List[Edge]) -> int:
        """Compute graph diameter (longest shortest path).
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            Graph diameter
        """
        G = GraphMetricsComputer._build_networkx_graph(nodes, edges)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
        return nx.diameter(G)
    
    @staticmethod
    def compute_modularity(nodes: List[Node], edges: List[Edge]) -> float:
        """Compute graph modularity using Louvain method.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            Modularity score
        """
        G = GraphMetricsComputer._build_networkx_graph(nodes, edges)
        communities = community_louvain.best_partition(G)
        return community_louvain.modularity(communities, G)
    
    @staticmethod
    def compute_bridge_nodes(nodes: List[Node], edges: List[Edge]) -> Tuple[Set[str], float]:
        """Identify bridge nodes and compute their ratio.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            Tuple of (bridge node IDs, bridge node ratio)
        """
        G = GraphMetricsComputer._build_networkx_graph(nodes, edges)
        
        # Get communities
        communities = community_louvain.best_partition(G)
        community_to_nodes = {}
        
        # Group nodes by community
        for node, community in communities.items():
            if community not in community_to_nodes:
                community_to_nodes[community] = set()
            community_to_nodes[community].add(node)
        
        # Find nodes with edges to multiple communities
        bridge_nodes = set()
        for node in G.nodes():
            node_community = communities[node]
            neighbor_communities = {communities[neighbor] 
                                 for neighbor in G.neighbors(node)}
            
            if len(neighbor_communities - {node_community}) > 0:
                bridge_nodes.add(node)
        
        bridge_ratio = len(bridge_nodes) / len(nodes) if nodes else 0.0
        return bridge_nodes, bridge_ratio
    
    @staticmethod
    def compute_metrics(nodes: List[Node], edges: List[Edge]) -> Dict[str, float]:
        """Compute all graph metrics.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            Dictionary of metric names to values
        """
        try:
            avg_path = GraphMetricsComputer.compute_avg_path_length(nodes, edges)
        except:
            avg_path = float('inf')
            
        try:
            diameter = float(GraphMetricsComputer.compute_diameter(nodes, edges))
        except:
            diameter = float('inf')
            
        try:
            modularity = GraphMetricsComputer.compute_modularity(nodes, edges)
        except:
            modularity = 0.0
            
        try:
            _, bridge_ratio = GraphMetricsComputer.compute_bridge_nodes(nodes, edges)
        except:
            bridge_ratio = 0.0
            
        return {
            "avg_path_length": avg_path,
            "diameter": diameter,
            "modularity": modularity,
            "bridge_node_ratio": bridge_ratio
        }
