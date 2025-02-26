"""Bridge node management for knowledge graph reasoning."""
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import networkx as nx
import logging
from collections import defaultdict, Counter
from datetime import datetime

log = logging.getLogger(__name__)


class BridgeNodeManager:
    """Bridge node management for knowledge graph reasoning.
    
    Implements persistence tracking over iterations, influence measurement,
    and cross-domain connector identification for bridge nodes.
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None):
        """Initialize bridge node manager.
        
        Args:
            graph: Optional NetworkX graph to analyze
        """
        self.graph = graph or nx.Graph()
        self.bridge_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.domain_mapping: Dict[str, str] = {}
        self.persistence_threshold = 3  # Number of iterations to consider a node persistent
        self.influence_threshold = 0.6  # Threshold for high influence
        self.cross_domain_threshold = 2  # Minimum number of domains to connect
    
    def set_graph(self, graph: nx.Graph) -> None:
        """Set the graph to analyze.
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
    
    def set_domain_mapping(self, domain_mapping: Dict[str, str]) -> None:
        """Set domain mapping for nodes.
        
        Args:
            domain_mapping: Mapping from node ID to domain
        """
        self.domain_mapping = domain_mapping
    
    async def identify_bridge_nodes(self) -> List[str]:
        """Identify bridge nodes connecting communities.
        
        Returns:
            List[str]: IDs of bridge nodes
        """
        try:
            if len(self.graph) < 3:
                return []
            
            # Find communities using Louvain method
            communities = nx.community.louvain_communities(self.graph)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Find nodes connecting different communities
            bridge_nodes = set()
            for node in self.graph.nodes():
                node_community = community_map.get(node)
                if node_community is None:
                    continue
                
                neighbor_communities = set()
                for neighbor in self.graph.neighbors(node):
                    neighbor_community = community_map.get(neighbor)
                    if neighbor_community is not None:
                        neighbor_communities.add(neighbor_community)
                
                # Node is a bridge if it connects to multiple communities
                if len(neighbor_communities) > 1:
                    bridge_nodes.add(node)
            
            # Sort by betweenness centrality
            centrality = nx.betweenness_centrality(self.graph)
            return sorted(
                bridge_nodes,
                key=lambda x: centrality.get(x, 0),
                reverse=True
            )
        except Exception as e:
            log.error(f"Failed to identify bridge nodes: {e}")
            return []
    
    async def compute_bridge_node_metrics(self, bridge_nodes: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compute metrics for bridge nodes.
        
        Args:
            bridge_nodes: List of bridge node IDs
            
        Returns:
            Dict[str, Dict[str, Any]]: Metrics for each bridge node
        """
        try:
            if not bridge_nodes or len(self.graph) < 3:
                return {}
            
            # Compute centrality measures
            betweenness = nx.betweenness_centrality(self.graph)
            degree = nx.degree_centrality(self.graph)
            
            try:
                eigenvector = nx.eigenvector_centrality(self.graph)
            except Exception:
                # Fall back to degree centrality if eigenvector fails
                eigenvector = degree
            
            # Find communities
            communities = nx.community.louvain_communities(self.graph)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Compute metrics for each bridge node
            metrics = {}
            for node in bridge_nodes:
                if node not in self.graph:
                    continue
                
                # Get connected communities
                node_community = community_map.get(node)
                if node_community is None:
                    continue
                
                connected_communities = set()
                for neighbor in self.graph.neighbors(node):
                    neighbor_community = community_map.get(neighbor)
                    if neighbor_community is not None and neighbor_community != node_community:
                        connected_communities.add(neighbor_community)
                
                # Get connected domains
                connected_domains = set()
                if self.domain_mapping:
                    node_domain = self.domain_mapping.get(node)
                    for neighbor in self.graph.neighbors(node):
                        neighbor_domain = self.domain_mapping.get(neighbor)
                        if neighbor_domain and neighbor_domain != node_domain:
                            connected_domains.add(neighbor_domain)
                
                # Compute influence score
                influence = (
                    0.5 * betweenness.get(node, 0) +
                    0.3 * eigenvector.get(node, 0) +
                    0.2 * degree.get(node, 0)
                )
                
                metrics[node] = {
                    "betweenness": betweenness.get(node, 0),
                    "eigenvector": eigenvector.get(node, 0),
                    "degree": degree.get(node, 0),
                    "influence": influence,
                    "connected_communities": len(connected_communities),
                    "connected_domains": len(connected_domains),
                    "is_cross_domain": len(connected_domains) >= self.cross_domain_threshold,
                    "timestamp": datetime.now().isoformat()
                }
            
            return metrics
        except Exception as e:
            log.error(f"Failed to compute bridge node metrics: {e}")
            return {}
    
    async def track_bridge_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Track bridge nodes and their metrics over time.
        
        Returns:
            Dict[str, Dict[str, Any]]: Current metrics for bridge nodes
        """
        try:
            # Identify bridge nodes
            bridge_nodes = await self.identify_bridge_nodes()
            
            # Compute metrics
            metrics = await self.compute_bridge_node_metrics(bridge_nodes)
            
            # Update history
            for node, node_metrics in metrics.items():
                self.bridge_history[node].append(node_metrics)
            
            return metrics
        except Exception as e:
            log.error(f"Failed to track bridge nodes: {e}")
            return {}
    
    async def get_persistent_bridge_nodes(self) -> List[str]:
        """Get bridge nodes that persist over multiple iterations.
        
        Returns:
            List[str]: IDs of persistent bridge nodes
        """
        try:
            persistent_nodes = []
            
            for node, history in self.bridge_history.items():
                if len(history) >= self.persistence_threshold:
                    # Check if node was a bridge in the last N iterations
                    recent_history = history[-self.persistence_threshold:]
                    if all(
                        metrics.get("connected_communities", 0) > 1
                        for metrics in recent_history
                    ):
                        persistent_nodes.append(node)
            
            return persistent_nodes
        except Exception as e:
            log.error(f"Failed to get persistent bridge nodes: {e}")
            return []
    
    async def get_high_influence_bridge_nodes(self) -> List[str]:
        """Get bridge nodes with high influence.
        
        Returns:
            List[str]: IDs of high influence bridge nodes
        """
        try:
            high_influence_nodes = []
            
            for node, history in self.bridge_history.items():
                if not history:
                    continue
                
                # Get latest metrics
                latest = history[-1]
                
                # Check influence
                if latest.get("influence", 0) >= self.influence_threshold:
                    high_influence_nodes.append(node)
            
            return high_influence_nodes
        except Exception as e:
            log.error(f"Failed to get high influence bridge nodes: {e}")
            return []
    
    async def get_cross_domain_bridge_nodes(self) -> List[str]:
        """Get bridge nodes that connect multiple domains.
        
        Returns:
            List[str]: IDs of cross-domain bridge nodes
        """
        try:
            if not self.domain_mapping:
                return []
            
            cross_domain_nodes = []
            
            for node, history in self.bridge_history.items():
                if not history:
                    continue
                
                # Get latest metrics
                latest = history[-1]
                
                # Check if connects multiple domains
                if latest.get("is_cross_domain", False):
                    cross_domain_nodes.append(node)
            
            return cross_domain_nodes
        except Exception as e:
            log.error(f"Failed to get cross-domain bridge nodes: {e}")
            return []
    
    async def analyze_bridge_node_evolution(self, node_id: str) -> Dict[str, Any]:
        """Analyze evolution of a bridge node over time.
        
        Args:
            node_id: Bridge node ID
            
        Returns:
            Dict[str, Any]: Evolution analysis
        """
        try:
            history = self.bridge_history.get(node_id, [])
            
            if not history:
                return {
                    "node_id": node_id,
                    "persistence": 0,
                    "stability": 0.0,
                    "influence_trend": "unknown",
                    "community_connections_trend": "unknown"
                }
            
            # Compute persistence
            persistence = len(history)
            
            # Compute stability (inverse of variance in influence)
            if len(history) > 1:
                influence_values = [h.get("influence", 0) for h in history]
                stability = 1.0 / (np.var(influence_values) + 0.0001)
                stability = min(stability, 100.0)  # Cap at 100
            else:
                stability = 0.0
            
            # Compute influence trend
            if len(history) > 2:
                recent_influence = [h.get("influence", 0) for h in history[-3:]]
                if recent_influence[-1] > recent_influence[0] * 1.1:
                    influence_trend = "increasing"
                elif recent_influence[-1] < recent_influence[0] * 0.9:
                    influence_trend = "decreasing"
                else:
                    influence_trend = "stable"
            else:
                influence_trend = "unknown"
            
            # Compute community connections trend
            if len(history) > 2:
                recent_connections = [
                    h.get("connected_communities", 0) for h in history[-3:]
                ]
                if recent_connections[-1] > recent_connections[0]:
                    community_trend = "increasing"
                elif recent_connections[-1] < recent_connections[0]:
                    community_trend = "decreasing"
                else:
                    community_trend = "stable"
            else:
                community_trend = "unknown"
            
            return {
                "node_id": node_id,
                "persistence": persistence,
                "stability": stability,
                "influence_trend": influence_trend,
                "community_connections_trend": community_trend,
                "is_persistent": persistence >= self.persistence_threshold,
                "is_high_influence": history[-1].get("influence", 0) >= self.influence_threshold,
                "is_cross_domain": history[-1].get("is_cross_domain", False)
            }
        except Exception as e:
            log.error(f"Failed to analyze bridge node evolution: {e}")
            return {
                "node_id": node_id,
                "persistence": 0,
                "stability": 0.0,
                "influence_trend": "unknown",
                "community_connections_trend": "unknown"
            }
    
    async def get_domain_bridge_analysis(self) -> Dict[str, Any]:
        """Analyze bridge nodes by domain.
        
        Returns:
            Dict[str, Any]: Domain bridge analysis
        """
        try:
            if not self.domain_mapping:
                return {
                    "domains": [],
                    "domain_connections": {},
                    "bridge_distribution": {}
                }
            
            # Get all domains
            domains = set(self.domain_mapping.values())
            
            # Count bridge nodes by domain
            domain_bridges: Dict[str, int] = Counter()
            
            # Track domain connections
            domain_connections: Dict[str, Set[str]] = defaultdict(set)
            
            # Analyze bridge nodes
            bridge_nodes = await self.identify_bridge_nodes()
            for node in bridge_nodes:
                node_domain = self.domain_mapping.get(node)
                if not node_domain:
                    continue
                
                domain_bridges[node_domain] += 1
                
                # Find connected domains
                for neighbor in self.graph.neighbors(node):
                    neighbor_domain = self.domain_mapping.get(neighbor)
                    if neighbor_domain and neighbor_domain != node_domain:
                        domain_connections[node_domain].add(neighbor_domain)
            
            # Convert sets to lists for JSON serialization
            domain_connections_list = {
                domain: list(connections)
                for domain, connections in domain_connections.items()
            }
            
            return {
                "domains": list(domains),
                "domain_connections": domain_connections_list,
                "bridge_distribution": dict(domain_bridges)
            }
        except Exception as e:
            log.error(f"Failed to get domain bridge analysis: {e}")
            return {
                "domains": [],
                "domain_connections": {},
                "bridge_distribution": {}
            }
    
    async def recommend_new_bridge_nodes(self) -> List[Dict[str, Any]]:
        """Recommend potential new bridge nodes.
        
        Returns:
            List[Dict[str, Any]]: Recommendations
        """
        try:
            if len(self.graph) < 5:
                return []
            
            # Find communities
            communities = nx.community.louvain_communities(self.graph)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Find existing bridge nodes
            existing_bridges = set(await self.identify_bridge_nodes())
            
            # Find potential bridges
            potential_bridges = []
            
            # Compute node centrality
            centrality = nx.betweenness_centrality(self.graph)
            
            # For each community pair, find nodes that could connect them
            for i, community1 in enumerate(communities):
                for j in range(i + 1, len(communities)):
                    community2 = communities[j]
                    
                    # Skip if communities are already well-connected
                    cross_edges = 0
                    for node1 in community1:
                        for node2 in community2:
                            if self.graph.has_edge(node1, node2):
                                cross_edges += 1
                    
                    # If communities are already well-connected, skip
                    if cross_edges > min(len(community1), len(community2)) / 4:
                        continue
                    
                    # Find high centrality nodes in each community
                    high_centrality1 = sorted(
                        [n for n in community1 if n not in existing_bridges],
                        key=lambda x: centrality.get(x, 0),
                        reverse=True
                    )[:3]
                    
                    high_centrality2 = sorted(
                        [n for n in community2 if n not in existing_bridges],
                        key=lambda x: centrality.get(x, 0),
                        reverse=True
                    )[:3]
                    
                    # Recommend connections between high centrality nodes
                    for node1 in high_centrality1:
                        for node2 in high_centrality2:
                            if not self.graph.has_edge(node1, node2):
                                potential_bridges.append({
                                    "source": node1,
                                    "target": node2,
                                    "source_community": i,
                                    "target_community": j,
                                    "source_centrality": centrality.get(node1, 0),
                                    "target_centrality": centrality.get(node2, 0),
                                    "score": centrality.get(node1, 0) + centrality.get(node2, 0)
                                })
            
            # Sort by score
            potential_bridges.sort(key=lambda x: x["score"], reverse=True)
            
            return potential_bridges[:10]  # Return top 10 recommendations
        except Exception as e:
            log.error(f"Failed to recommend new bridge nodes: {e}")
            return []
    
    async def get_bridge_node_summary(self) -> Dict[str, Any]:
        """Get summary of bridge node analysis.
        
        Returns:
            Dict[str, Any]: Summary
        """
        try:
            # Identify bridge nodes
            bridge_nodes = await self.identify_bridge_nodes()
            
            # Get persistent bridge nodes
            persistent_nodes = await self.get_persistent_bridge_nodes()
            
            # Get high influence bridge nodes
            high_influence_nodes = await self.get_high_influence_bridge_nodes()
            
            # Get cross-domain bridge nodes
            cross_domain_nodes = await self.get_cross_domain_bridge_nodes()
            
            # Get domain analysis
            domain_analysis = await self.get_domain_bridge_analysis()
            
            # Get recommendations
            recommendations = await self.recommend_new_bridge_nodes()
            
            return {
                "total_bridge_nodes": len(bridge_nodes),
                "persistent_bridge_nodes": len(persistent_nodes),
                "high_influence_bridge_nodes": len(high_influence_nodes),
                "cross_domain_bridge_nodes": len(cross_domain_nodes),
                "bridge_nodes": bridge_nodes,
                "persistent_nodes": persistent_nodes,
                "high_influence_nodes": high_influence_nodes,
                "cross_domain_nodes": cross_domain_nodes,
                "domain_analysis": domain_analysis,
                "recommendations": recommendations
            }
        except Exception as e:
            log.error(f"Failed to get bridge node summary: {e}")
            return {
                "total_bridge_nodes": 0,
                "persistent_bridge_nodes": 0,
                "high_influence_bridge_nodes": 0,
                "cross_domain_bridge_nodes": 0,
                "bridge_nodes": [],
                "persistent_nodes": [],
                "high_influence_nodes": [],
                "cross_domain_nodes": [],
                "domain_analysis": {
                    "domains": [],
                    "domain_connections": {},
                    "bridge_distribution": {}
                },
                "recommendations": []
            }
