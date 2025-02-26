"""Milvus implementation of the vector store."""
from typing import List, Optional, Dict, AsyncIterator
import asyncio
import json
import numpy as np
from pymilvus import MilvusClient, DataType  # type: ignore

from .base import BaseVectorStore
from .exceptions import MilvusError, CollectionInitError, SearchError
from ..models.node import Node
from ..models.edge import Edge


class MilvusStore(BaseVectorStore):
    """Milvus-based implementation of the vector store."""
    
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        dim: int = 1536,  # Default for deepseek-r1-671b
        default_collection: str = "knowledge_graph",
        token: str = "",
        db: str = "default",
    ):
        """Initialize the Milvus graph store.
        
        Args:
            uri: Milvus server URI
            dim: Dimension of the embeddings
            default_collection: Name of the default collection
            token: Authentication token
            db: Database name
        """
        self.client = MilvusClient(uri=uri, token=token, db_name=db, timeout=30)
        self.dim = dim
        self.default_collection = default_collection
        self._collections_initialized = False
    
    async def initialize(self) -> None:
        """Initialize collections. Must be called before using the store."""
        if self._collections_initialized:
            return
            
        # Create nodes collection if it doesn't exist
        if not self.client.has_collection(f"{self.default_collection}_nodes"):
            node_schema = self.client.create_schema(
                enable_dynamic_field=True,  # Enable dynamic fields for flexibility
                auto_id=False,  # We manage IDs
                description="Knowledge graph nodes"
            )
            node_schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256, auto_id=False)
            node_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
            node_schema.add_field("content", DataType.VARCHAR, max_length=65535)
            node_schema.add_field("metadata", DataType.JSON)
            
            collection_name = f"{self.default_collection}_nodes"
            self.client.create_collection(
                collection_name,
                schema=node_schema,
                consistency_level="Strong"
            )
            
            # Create collection with index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type="L2"
            )
            self.client.create_collection(
                collection_name=collection_name,
                schema=node_schema,
                index_params=index_params,
                consistency_level="Strong"
            )
        
        # Create edges collection if it doesn't exist
        if not self.client.has_collection(f"{self.default_collection}_edges"):
            edge_schema = self.client.create_schema(
                enable_dynamic_field=True,  # Enable dynamic fields for flexibility
                auto_id=False,  # We manage IDs
                description="Knowledge graph edges"
            )
            edge_schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256, auto_id=False)
            edge_schema.add_field("source", DataType.VARCHAR, max_length=256)
            edge_schema.add_field("target", DataType.VARCHAR, max_length=256)
            edge_schema.add_field("type", DataType.VARCHAR, max_length=256)
            edge_schema.add_field("metadata", DataType.JSON)
            
            self.client.create_collection(
                f"{self.default_collection}_edges",
                schema=edge_schema,
                consistency_level="Strong"
            )
            
        self._collections_initialized = True
    
    async def add_node(self, node: Node) -> str:
        """Add a node to the vector store."""
        try:
            data = {
                "id": node.id,
                "content": node.content,
                "metadata": node.metadata
            }
            self.client.insert(
                collection_name=f"{self.default_collection}_nodes",
                data=[data]
            )
            return node.id
        except Exception as e:
            raise MilvusError(f"Failed to add node: {e}")
    
    async def add_edge(self, edge: Edge) -> None:
        """Add an edge to the vector store."""
        try:
            data = {
                "id": f"{edge.source}_{edge.target}_{edge.type}",
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "metadata": edge.metadata
            }
            self.client.insert(
                collection_name=f"{self.default_collection}_edges",
                data=[data]
            )
        except Exception as e:
            raise MilvusError(f"Failed to add edge: {e}")
    
    async def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.5,
        deduplicate: bool = True
    ) -> List[Node]:
        """Search for similar nodes using L2 distance."""
        try:
            collection_name = f"{self.default_collection}_nodes"
            self.client.load_collection(collection_name)
            
            # Execute search with proper parameters
            results = self.client.search(
                collection_name=collection_name,
                data=[embedding.tolist()],
                anns_field="embedding",
                params={"nprobe": 10},
                limit=k,
                output_fields=["*"],
                metric_type="L2",
                consistency_level="Strong",
                timeout=10
            )
            
            nodes = []
            max_l2_dist = np.sqrt(2 * (1 - threshold))  # Convert similarity threshold to L2 distance
            
            for result_group in results:
                for hit in result_group:
                    if isinstance(hit.distance, (int, float)) and hit.distance <= max_l2_dist:
                        nodes.append(Node(
                            id=hit.entity["id"],
                            content=hit.entity["content"],
                            metadata=hit.entity["metadata"]
                        ))
            return nodes
        except Exception as e:
            raise SearchError(f"Search failed: {e}")
    
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        try:
            results = self.client.query(
                collection_name=f"{self.default_collection}_nodes",
                filter=f'id == "{node_id}"',
                output_fields=["*"]
            )
            if not results:
                return None
            
            entity = results[0]
            return Node(
                id=entity["id"],
                content=entity["content"],
                metadata=entity["metadata"]
            )
        except Exception as e:
            raise MilvusError(f"Failed to get node: {e}")
    
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
            AsyncIterator[Edge]: Matching edges
        """
        try:
            filters = []
            if source_id:
                filters.append(f'source == "{source_id}"')
            if target_id:
                filters.append(f'target == "{target_id}"')
            if edge_type:
                filters.append(f'type == "{edge_type}"')
            
            expr = " && ".join(filters) if filters else ""
            
            results = self.client.query(
                collection_name=f"{self.default_collection}_edges",
                filter=expr if expr else None,
                output_fields=["*"]
            )
            
            for entity in results:
                yield Edge(
                    source=entity["source"],
                    target=entity["target"],
                    type=entity["type"],
                    metadata=entity["metadata"]
                )
                await asyncio.sleep(0)  # Allow other coroutines to run
        except Exception as e:
            raise MilvusError(f"Failed to get edges: {e}")
    
    async def update_node(self, node: Node) -> None:
        """Update a node in the vector store.
        
        Args:
            node: Node to update
        """
        await self.initialize()
        
        try:
            # Check if node exists
            existing = await self.get_node(node.id)
            if not existing:
                raise MilvusError(f"Node with ID {node.id} not found")
            
            # Delete existing node
            self.client.delete(
                collection_name=f"{self.default_collection}_nodes",
                filter=f'id == "{node.id}"'
            )
            
            # Insert updated node
            data = {
                "id": node.id,
                "content": node.content,
                "metadata": node.metadata
            }
            
            # Add embedding if available
            if "embedding" in node.metadata:
                data["embedding"] = node.metadata["embedding"]
            
            self.client.insert(
                collection_name=f"{self.default_collection}_nodes",
                data=[data]
            )
        except Exception as e:
            raise MilvusError(f"Failed to update node: {e}")
    
    async def get_all_nodes(self) -> AsyncIterator[Node]:
        """Get all nodes in the vector store.
        
        Returns:
            AsyncIterator[Node]: All nodes
        """
        async def _get_all_nodes():
            await self.initialize()
            
            try:
                # Query all nodes
                results = self.client.query(
                    collection_name=f"{self.default_collection}_nodes",
                    filter=None,  # No filter means all nodes
                    output_fields=["*"]
                )
                
                for entity in results:
                    yield Node(
                        id=entity["id"],
                        content=entity["content"],
                        metadata=entity["metadata"]
                    )
                    await asyncio.sleep(0)  # Allow other coroutines to run
            except Exception as e:
                raise MilvusError(f"Failed to get all nodes: {e}")
                
        return _get_all_nodes()
    
    async def get_all_edges(self) -> AsyncIterator[Edge]:
        """Get all edges in the vector store.
        
        Returns:
            AsyncIterator[Edge]: All edges
        """
        async def _get_all_edges():
            await self.initialize()
            
            try:
                # Query all edges
                results = self.client.query(
                    collection_name=f"{self.default_collection}_edges",
                    filter=None,  # No filter means all edges
                    output_fields=["*"]
                )
                
                for entity in results:
                    yield Edge(
                        source=entity["source"],
                        target=entity["target"],
                        type=entity["type"],
                        metadata=entity["metadata"]
                    )
                    await asyncio.sleep(0)  # Allow other coroutines to run
            except Exception as e:
                raise MilvusError(f"Failed to get all edges: {e}")
                
        return _get_all_edges()
