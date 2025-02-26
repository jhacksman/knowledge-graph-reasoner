"""Integration module for bridge node management."""
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import networkx as nx
from datetime import datetime

from .bridge_node_manager import BridgeNodeManager
from ..metrics.metrics import GraphMetrics

log = logging.getLogger(__name__)


class BridgeNodeIntegration:
    """Integration of bridge node management with the reasoning pipeline."""
    
    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        domain_mapping: Optional[Dict[str, str]] = None,
        persistence_threshold: int = 3,
        influence_threshold: float = 0.6,
        cross_domain_threshold: int = 2
    ):
        """Initialize bridge node integration.
        
        Args:
            graph: Optional NetworkX graph to analyze
            domain_mapping: Optional mapping from node ID to domain
            persistence_threshold: Number of iterations to consider a node persistent
            influence_threshold: Threshold for high influence
            cross_domain_threshold: Minimum number of domains to connect
        """
        self.manager = BridgeNodeManager(graph)
        
        if domain_mapping:
            self.manager.set_domain_mapping(domain_mapping)
        
        self.manager.persistence_threshold = persistence_threshold
        self.manager.influence_threshold = influence_threshold
        self.manager.cross_domain_threshold = cross_domain_threshold
        
        self.metrics = GraphMetrics(graph) if graph else GraphMetrics()
    
    def set_graph(self, graph: nx.Graph) -> None:
        """Set the graph to analyze.
        
        Args:
            graph: NetworkX graph
        """
        self.manager.set_graph(graph)
        self.metrics.set_graph(graph)
    
    def set_domain_mapping(self, domain_mapping: Dict[str, str]) -> None:
        """Set domain mapping for nodes.
        
        Args:
            domain_mapping: Mapping from node ID to domain
        """
        self.manager.set_domain_mapping(domain_mapping)
    
    async def update_bridge_nodes(self) -> Dict[str, Any]:
        """Update bridge node tracking and return summary.
        
        Returns:
            Dict[str, Any]: Bridge node summary
        """
        # Track bridge nodes
        await self.manager.track_bridge_nodes()
        
        # Get summary
        return await self.manager.get_bridge_node_summary()
    
    async def get_bridge_node_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for new bridge nodes.
        
        Returns:
            List[Dict[str, Any]]: Recommendations
        """
        return await self.manager.recommend_new_bridge_nodes()
    
    async def get_bridge_node_evolution(self, node_id: str) -> Dict[str, Any]:
        """Get evolution analysis for a bridge node.
        
        Args:
            node_id: Bridge node ID
            
        Returns:
            Dict[str, Any]: Evolution analysis
        """
        return await self.manager.analyze_bridge_node_evolution(node_id)
    
    async def get_domain_bridge_analysis(self) -> Dict[str, Any]:
        """Get domain bridge analysis.
        
        Returns:
            Dict[str, Any]: Domain bridge analysis
        """
        return await self.manager.get_domain_bridge_analysis()
    
    async def get_persistent_bridge_nodes(self) -> List[str]:
        """Get persistent bridge nodes.
        
        Returns:
            List[str]: Persistent bridge nodes
        """
        return await self.manager.get_persistent_bridge_nodes()
    
    async def get_high_influence_bridge_nodes(self) -> List[str]:
        """Get high influence bridge nodes.
        
        Returns:
            List[str]: High influence bridge nodes
        """
        return await self.manager.get_high_influence_bridge_nodes()
    
    async def get_cross_domain_bridge_nodes(self) -> List[str]:
        """Get cross-domain bridge nodes.
        
        Returns:
            List[str]: Cross-domain bridge nodes
        """
        return await self.manager.get_cross_domain_bridge_nodes()
    
    async def get_bridge_metrics(self) -> Dict[str, Any]:
        """Get combined bridge metrics.
        
        Returns:
            Dict[str, Any]: Combined metrics
        """
        try:
            # Get bridge nodes
            bridge_nodes = await self.manager.identify_bridge_nodes()
            
            # Get graph metrics
            graph_metrics = await self.metrics.compute_metrics()
            
            # Get bridge node summary
            bridge_summary = await self.manager.get_bridge_node_summary()
            
            # Get domain analysis
            domain_analysis = await self.manager.get_domain_bridge_analysis()
            
            # Combine metrics
            combined_metrics = {
                "timestamp": datetime.now().isoformat(),
                "graph_metrics": graph_metrics,
                "bridge_summary": bridge_summary,
                "domain_analysis": domain_analysis,
                "recommendations": await self.manager.recommend_new_bridge_nodes()
            }
            
            return combined_metrics
        except Exception as e:
            log.error(f"Failed to get bridge metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "graph_metrics": {},
                "bridge_summary": {},
                "domain_analysis": {},
                "recommendations": []
            }
