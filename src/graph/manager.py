"""Graph manager implementation."""
from typing import Dict, Any, List, Optional
import uuid
import numpy as np
import logging

from ..vector_store.base import BaseVectorStore
from ..models.node import Node
from ..models.edge import Edge
from ..metrics.graph_metrics import GraphMetrics

log = logging.getLogger(__name__)


class GraphManager:
    """Manages knowledge graph operations and state."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        metrics: Optional[GraphMetrics] = None
    ):
        """Initialize graph manager.
        
        Args:
            vector_store: Vector store for graph storage
            metrics: Optional graph metrics tracker
        """
        self.vector_store = vector_store
        self.metrics = metrics or GraphMetrics()
    
    async def add_concept(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add new concept to graph.
        
        Args:
            content: Concept text content
            embedding: Vector embedding
            metadata: Optional metadata
            
        Returns:
            str: ID of added concept
        """
        try:
            node = Node(
                id=str(uuid.uuid4()),
                content=content,
                metadata={
                    "embedding": embedding.tolist(),
                    **(metadata or {})
                }
            )
            return await self.vector_store.add_node(node)
        except Exception as e:
            log.error(f"Failed to add concept: {e}")
            raise
    
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add relationship between concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship
            metadata: Optional metadata
        """
        try:
            edge = Edge(
                source=source_id,
                target=target_id,
                type=relationship_type,
                metadata=metadata or {}
            )
            await self.vector_store.add_edge(edge)
        except Exception as e:
            log.error(f"Failed to add relationship: {e}")
            raise
    
    async def get_similar_concepts(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.5
    ) -> List[Node]:
        """Find similar concepts by embedding.
        
        Args:
            embedding: Query embedding
            k: Number of results
            threshold: Similarity threshold
            
        Returns:
            List[Node]: Similar concepts
        """
        try:
            return await self.vector_store.search_similar(
                embedding,
                k=k,
                threshold=threshold
            )
        except Exception as e:
            log.error(f"Failed to find similar concepts: {e}")
            raise
    
    async def get_concept(self, concept_id: str) -> Optional[Node]:
        """Get concept by ID.
        
        Args:
            concept_id: Concept ID
            
        Returns:
            Optional[Node]: Concept if found
        """
        try:
            return await self.vector_store.get_node(concept_id)
        except Exception as e:
            log.error(f"Failed to get concept: {e}")
            raise
    
    async def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[str] = None
    ) -> List[Edge]:
        """Get relationships matching criteria.
        
        Args:
            source_id: Optional source concept ID
            target_id: Optional target concept ID
            relationship_type: Optional relationship type
            
        Returns:
            List[Edge]: Matching relationships
        """
        try:
            edges = []
            async for edge in self.vector_store.get_edges(
                source_id=source_id,
                target_id=target_id,
                edge_type=relationship_type
            ):
                edges.append(edge)
            return edges
        except Exception as e:
            log.error(f"Failed to get relationships: {e}")
            raise
    
    async def get_graph_state(self) -> Dict[str, Any]:
        """Get current graph metrics.
        
        Returns:
            Dict[str, Any]: Graph metrics including:
                - modularity
                - avg_path_length
                - bridge_nodes
                - diameter
        """
        try:
            return {
                "modularity": await self.metrics.compute_modularity(),
                "avg_path_length": await self.metrics.compute_avg_path_length(),
                "bridge_nodes": await self.metrics.find_bridge_nodes(),
                "diameter": await self.metrics.compute_diameter()
            }
        except Exception as e:
            log.error(f"Failed to get graph state: {e}")
            raise
