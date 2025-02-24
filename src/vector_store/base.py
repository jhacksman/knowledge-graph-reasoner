"""Base interfaces for vector store implementations."""
from typing import List, Optional, Dict, AsyncIterator
import numpy as np
from abc import ABC, abstractmethod

from ..models.node import Node
from ..models.edge import Edge


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    async def add_node(self, node: Node) -> str:
        """Add a node to the vector store.
        
        Args:
            node: Node to add
            
        Returns:
            str: ID of the added node
        """
        raise NotImplementedError
    
    @abstractmethod
    async def add_edge(self, edge: Edge) -> None:
        """Add an edge to the vector store.
        
        Args:
            edge: Edge to add
        """
        raise NotImplementedError
    
    @abstractmethod
    async def search_similar(
        self, 
        embedding: np.ndarray, 
        k: int = 5,
        threshold: float = 0.0
    ) -> List[Node]:
        """Search for nodes with similar embeddings.
        
        Args:
            embedding: Query embedding
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List[Node]: List of similar nodes
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Optional[Node]: The node if found, None otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_edges(
        self, 
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None
    ) -> AsyncIterator[Edge]:
        """Get edges matching the given criteria.
        
        Args:
            source_id: Optional source node ID filter
            target_id: Optional target node ID filter
            edge_type: Optional edge type filter
            
        Returns:
            AsyncIterator[Edge]: Iterator over matching edges
        """
        raise NotImplementedError
