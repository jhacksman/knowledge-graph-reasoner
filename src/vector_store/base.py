"""Base class for vector store implementations."""
from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator, Set
import numpy as np

from ..models.node import Node
from ..models.edge import Edge


def deduplicate_nodes(nodes: List[Node]) -> List[Node]:
    """Deduplicate nodes based on content.
    
    Args:
        nodes: List of nodes to deduplicate
        
    Returns:
        List[Node]: Deduplicated nodes, keeping first occurrence of each content
    """
    seen_content: Set[str] = set()
    unique_nodes: List[Node] = []
    
    for node in nodes:
        if node.content not in seen_content:
            seen_content.add(node.content)
            unique_nodes.append(node)
    
    return unique_nodes


def deduplicate_edges(edges: List[Edge]) -> List[Edge]:
    """Deduplicate edges based on source, target and type.
    
    Args:
        edges: List of edges to deduplicate
        
    Returns:
        List[Edge]: Deduplicated edges
    """
    seen_keys: Set[str] = set()
    unique_edges: List[Edge] = []
    
    for edge in edges:
        # Create unique key from source, target and type
        edge_key = f"{edge.source}_{edge.target}_{edge.type}"
        
        if edge_key not in seen_keys:
            seen_keys.add(edge_key)
            unique_edges.append(edge)
    
    return unique_edges


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def add_node(self, node: Node) -> str:
        """Add a node to the vector store.
        
        Args:
            node: Node to add
            
        Returns:
            str: ID of the added node
        """
        pass
    
    @abstractmethod
    async def add_edge(self, edge: Edge) -> None:
        """Add an edge to the vector store.
        
        Args:
            edge: Edge to add
        """
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.5,
        deduplicate: bool = True
    ) -> List[Node]:
        """Search for similar nodes.
        
        Args:
            embedding: Query embedding
            k: Number of results to return
            threshold: Similarity threshold (0-1)
            deduplicate: Whether to deduplicate results
            
        Returns:
            List[Node]: Similar nodes
        """
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID.
        
        Args:
            node_id: ID of the node to get
            
        Returns:
            Optional[Node]: The node if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_edges(
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
            AsyncIterator[Edge]: Matching edges
        """
        pass
