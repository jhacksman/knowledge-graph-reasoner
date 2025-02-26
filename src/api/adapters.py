"""Adapter functions for converting between API models and internal models."""
from typing import List, Dict, Any, Optional
import numpy as np

from src.models.node import Node
from src.models.edge import Edge
from src.api.models import Concept, ConceptCreate, ConceptUpdate


def node_to_concept(node: Node) -> Concept:
    """Convert a Node to a Concept."""
    # Extract embedding from metadata if present
    embedding = None
    if node.metadata and "embedding" in node.metadata:
        embedding = node.metadata["embedding"]
    
    # Create concept from node
    return Concept(
        id=node.id,
        content=node.content,
        embedding=embedding,
        metadata={k: v for k, v in (node.metadata or {}).items() if k != "embedding"}
    )


def nodes_to_concepts(nodes: List[Node]) -> List[Concept]:
    """Convert a list of Nodes to a list of Concepts."""
    return [node_to_concept(node) for node in nodes]


def concept_to_node(concept: ConceptCreate) -> Dict[str, Any]:
    """Convert a ConceptCreate to Node parameters."""
    # Prepare metadata
    metadata = concept.metadata or {}
    
    # Return node parameters
    return {
        "content": concept.content,
        "embedding": concept.embedding,
        "metadata": metadata
    }


def concept_update_to_node_update(concept: ConceptUpdate, concept_id: str) -> Dict[str, Any]:
    """Convert a ConceptUpdate to Node update parameters."""
    # Prepare update parameters
    update = {}
    
    if concept.content is not None:
        update["content"] = concept.content
    
    if concept.embedding is not None:
        update["embedding"] = concept.embedding
    
    if concept.metadata is not None:
        update["metadata"] = concept.metadata
    
    return update
