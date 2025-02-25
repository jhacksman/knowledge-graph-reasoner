"""Graph manager implementation."""
from typing import Dict, Any, List, Optional, AsyncIterator, Tuple, Union, Set
import asyncio
import uuid
import numpy as np
import logging
import json
import pickle
import gzip
import os
from pathlib import Path

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
            async for edge in aiter(self.vector_store.get_edges(
                source_id=source_id,
                target_id=target_id,
                edge_type=relationship_type
            )):
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
    
    async def export_graph_structure(self, path: Union[str, Path], compress: bool = False) -> Path:
        """Export graph structure to file.
        
        Args:
            path: Path to save the graph structure
            compress: Whether to compress the output
            
        Returns:
            Path: Path to the saved file
        """
        try:
            path = Path(path)
            
            # Get all nodes
            nodes = []
            async for node in self.vector_store.get_all_nodes():
                # Remove embedding from metadata to avoid duplication
                node_dict = node.dict()
                if "embedding" in node_dict["metadata"]:
                    del node_dict["metadata"]["embedding"]
                nodes.append(node_dict)
            
            # Get all edges
            edges = []
            async for edge in self.vector_store.get_all_edges():
                edges.append(edge.dict())
            
            # Create graph structure
            graph_structure = {
                "nodes": nodes,
                "edges": edges
            }
            
            # Save to file
            if compress:
                with gzip.open(f"{path}.gz", "wt") as f:
                    json.dump(graph_structure, f)
                return Path(f"{path}.gz")
            else:
                with open(path, "w") as f:
                    json.dump(graph_structure, f)
                return path
                
        except Exception as e:
            log.error(f"Failed to export graph structure: {e}")
            raise
    
    async def export_embeddings(self, path: Union[str, Path], compress: bool = True) -> Path:
        """Export embeddings to file.
        
        Args:
            path: Path to save the embeddings
            compress: Whether to compress the output
            
        Returns:
            Path: Path to the saved file
        """
        try:
            path = Path(path)
            
            # Get all nodes with embeddings
            embeddings = {}
            async for node in self.vector_store.get_all_nodes():
                if "embedding" in node.metadata:
                    embeddings[node.id] = np.array(node.metadata["embedding"])
            
            # Save to file
            if compress:
                with gzip.open(f"{path}.gz", "wb") as f:
                    pickle.dump(embeddings, f)
                return Path(f"{path}.gz")
            else:
                with open(path, "wb") as f:
                    pickle.dump(embeddings, f)
                return path
                
        except Exception as e:
            log.error(f"Failed to export embeddings: {e}")
            raise
    
    async def export_metrics_history(self, path: Union[str, Path]) -> Path:
        """Export metrics history to file.
        
        Args:
            path: Path to save the metrics history
            
        Returns:
            Path: Path to the saved file
        """
        try:
            path = Path(path)
            
            # Get metrics history
            metrics_history = await self.metrics.get_history()
            
            # Save to file
            with open(path, "w") as f:
                json.dump(metrics_history, f)
            
            return path
                
        except Exception as e:
            log.error(f"Failed to export metrics history: {e}")
            raise
    
    async def export_full_graph(
        self,
        directory: Union[str, Path],
        compress: bool = True,
        include_embeddings: bool = True,
        include_metrics: bool = True
    ) -> Dict[str, Path]:
        """Export full graph to directory.
        
        Args:
            directory: Directory to save the graph
            compress: Whether to compress the output
            include_embeddings: Whether to include embeddings
            include_metrics: Whether to include metrics history
            
        Returns:
            Dict[str, Path]: Paths to the saved files
        """
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            
            result = {}
            
            # Export graph structure
            structure_path = directory / "graph_structure.json"
            result["structure"] = await self.export_graph_structure(structure_path, compress)
            
            # Export embeddings if requested
            if include_embeddings:
                embeddings_path = directory / "embeddings.pkl"
                result["embeddings"] = await self.export_embeddings(embeddings_path, compress)
            
            # Export metrics history if requested
            if include_metrics:
                metrics_path = directory / "metrics_history.json"
                result["metrics"] = await self.export_metrics_history(metrics_path)
            
            return result
                
        except Exception as e:
            log.error(f"Failed to export full graph: {e}")
            raise
    
    async def import_graph_structure(self, path: Union[str, Path]) -> Tuple[int, int]:
        """Import graph structure from file.
        
        Args:
            path: Path to the graph structure file
            
        Returns:
            Tuple[int, int]: Number of nodes and edges imported
        """
        try:
            path = Path(path)
            
            # Load from file
            if str(path).endswith(".gz"):
                with gzip.open(path, "rt") as f:
                    graph_structure = json.load(f)
            else:
                with open(path, "r") as f:
                    graph_structure = json.load(f)
            
            # Import nodes
            node_count = 0
            for node_dict in graph_structure["nodes"]:
                node = Node(**node_dict)
                await self.vector_store.add_node(node)
                node_count += 1
            
            # Import edges
            edge_count = 0
            for edge_dict in graph_structure["edges"]:
                edge = Edge(**edge_dict)
                await self.vector_store.add_edge(edge)
                edge_count += 1
            
            return node_count, edge_count
                
        except Exception as e:
            log.error(f"Failed to import graph structure: {e}")
            raise
    
    async def import_embeddings(self, path: Union[str, Path]) -> int:
        """Import embeddings from file.
        
        Args:
            path: Path to the embeddings file
            
        Returns:
            int: Number of embeddings imported
        """
        try:
            path = Path(path)
            
            # Load from file
            if str(path).endswith(".gz"):
                with gzip.open(path, "rb") as f:
                    embeddings = pickle.load(f)
            else:
                with open(path, "rb") as f:
                    embeddings = pickle.load(f)
            
            # Update nodes with embeddings
            count = 0
            for node_id, embedding in embeddings.items():
                node = await self.get_concept(node_id)
                if node:
                    node.metadata["embedding"] = embedding.tolist()
                    await self.vector_store.update_node(node)
                    count += 1
            
            return count
                
        except Exception as e:
            log.error(f"Failed to import embeddings: {e}")
            raise
    
    async def import_metrics_history(self, path: Union[str, Path]) -> int:
        """Import metrics history from file.
        
        Args:
            path: Path to the metrics history file
            
        Returns:
            int: Number of metrics imported
        """
        try:
            path = Path(path)
            
            # Load from file
            with open(path, "r") as f:
                metrics_history = json.load(f)
            
            # Set metrics history
            await self.metrics.set_history(metrics_history)
            
            return len(metrics_history)
                
        except Exception as e:
            log.error(f"Failed to import metrics history: {e}")
            raise
    
    async def import_full_graph(
        self,
        directory: Union[str, Path],
        import_structure: bool = True,
        import_embeddings: bool = True,
        import_metrics: bool = True
    ) -> Dict[str, int]:
        """Import full graph from directory.
        
        Args:
            directory: Directory containing the graph files
            import_structure: Whether to import graph structure
            import_embeddings: Whether to import embeddings
            import_metrics: Whether to import metrics history
            
        Returns:
            Dict[str, int]: Number of items imported by type
        """
        try:
            directory = Path(directory)
            result = {}
            
            # Import graph structure if requested
            if import_structure:
                structure_path = directory / "graph_structure.json"
                if not structure_path.exists():
                    structure_path = directory / "graph_structure.json.gz"
                
                if structure_path.exists():
                    node_count, edge_count = await self.import_graph_structure(structure_path)
                    result["nodes"] = node_count
                    result["edges"] = edge_count
            
            # Import embeddings if requested
            if import_embeddings:
                embeddings_path = directory / "embeddings.pkl"
                if not embeddings_path.exists():
                    embeddings_path = directory / "embeddings.pkl.gz"
                
                if embeddings_path.exists():
                    embedding_count = await self.import_embeddings(embeddings_path)
                    result["embeddings"] = embedding_count
            
            # Import metrics history if requested
            if import_metrics:
                metrics_path = directory / "metrics_history.json"
                
                if metrics_path.exists():
                    metrics_count = await self.import_metrics_history(metrics_path)
                    result["metrics"] = metrics_count
            
            return result
                
        except Exception as e:
            log.error(f"Failed to import full graph: {e}")
            raise
