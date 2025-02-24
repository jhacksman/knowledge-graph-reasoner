"""Milvus implementation of the vector store."""
from typing import List, Optional, Dict, AsyncIterator
import numpy as np
from pymilvus import MilvusClient, DataType

from .base import BaseVectorStore
from ..models.node import Node
from ..models.edge import Edge


class MilvusGraphStore(BaseVectorStore):
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
        self._ensure_collections()
    
    def _ensure_collections(self) -> None:
        """Ensure required collections exist."""
        # Create nodes collection if it doesn't exist
        if not self.client.has_collection(f"{self.default_collection}_nodes"):
            node_schema = self.client.create_schema(
                enable_dynamic_field=False,
                auto_id=True,
                description="Knowledge graph nodes"
            )
            node_schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
            node_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
            node_schema.add_field("content", DataType.VARCHAR, max_length=65535)
            node_schema.add_field("metadata", DataType.JSON)
            
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE"  # Using cosine similarity for semantic search
            )
            
            self.client.create_collection(
                f"{self.default_collection}_nodes",
                schema=node_schema,
                index_params=index_params,
                consistency_level="Strong"
            )
        
        # Create edges collection if it doesn't exist
        if not self.client.has_collection(f"{self.default_collection}_edges"):
            edge_schema = self.client.create_schema(
                enable_dynamic_field=False,
                auto_id=True,
                description="Knowledge graph edges"
            )
            edge_schema.add_field("source", DataType.VARCHAR, max_length=256)
            edge_schema.add_field("target", DataType.VARCHAR, max_length=256)
            edge_schema.add_field("type", DataType.VARCHAR, max_length=256)
            edge_schema.add_field("metadata", DataType.JSON)
            
            self.client.create_collection(
                f"{self.default_collection}_edges",
                schema=edge_schema,
                consistency_level="Strong"
            )
    
    async def add_node(self, node: Node) -> str:
        """Add a node to the vector store."""
        try:
            data = {
                "id": node.id,
                "embedding": node.embedding.tolist(),
                "content": node.content,
                "metadata": node.metadata
            }
            self.client.insert(
                collection_name=f"{self.default_collection}_nodes",
                data=[data]
            )
            return node.id
        except Exception as e:
            raise RuntimeError(f"Failed to add node: {e}")
    
    async def add_edge(self, edge: Edge) -> None:
        """Add an edge to the vector store."""
        try:
            data = {
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
            raise RuntimeError(f"Failed to add edge: {e}")
    
    async def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[Node]:
        """Search for similar nodes using cosine similarity."""
        try:
            results = self.client.search(
                collection_name=f"{self.default_collection}_nodes",
                data=[embedding.tolist()],
                limit=k,
                output_fields=["*"],
                timeout=10
            )
            
            nodes = []
            for result_group in results:
                for hit in result_group:
                    if hit["distance"] >= threshold:  # Higher cosine similarity is better
                        nodes.append(Node(
                            id=hit["entity"]["id"],
                            embedding=np.array(hit["entity"]["embedding"]),
                            content=hit["entity"]["content"],
                            metadata=hit["entity"]["metadata"]
                        ))
            return nodes
        except Exception as e:
            raise RuntimeError(f"Failed to search nodes: {e}")
    
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
                embedding=np.array(entity["embedding"]),
                content=entity["content"],
                metadata=entity["metadata"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get node: {e}")
    
    async def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None
    ) -> AsyncIterator[Edge]:
        """Get edges matching the given criteria."""
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
        except Exception as e:
            raise RuntimeError(f"Failed to get edges: {e}")
