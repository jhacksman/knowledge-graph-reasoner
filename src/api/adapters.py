"""Adapter functions for converting between API models and internal models."""
from typing import List, Dict, Any, Optional, Union
import numpy as np

from src.models.node import Node
from src.models.edge import Edge
from src.api.models import Concept, ConceptCreate, ConceptUpdate, Relationship, RelationshipCreate, RelationshipUpdate


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


def map_relationship_params(
    relationship: Union[Relationship, RelationshipCreate, RelationshipUpdate]
) -> Dict[str, Any]:
    """Map relationship parameters to GraphManager method parameters.
    
    This function standardizes parameter mapping between API models and GraphManager methods.
    API models use 'type' for relationship type while GraphManager uses 'relationship_type'.
    
    Args:
        relationship: A Relationship, RelationshipCreate, or RelationshipUpdate object
        
    Returns:
        Dict with GraphManager method parameter names
    """
    return {
        "source_id": getattr(relationship, "source_id", None),
        "target_id": getattr(relationship, "target_id", None),
        "relationship_type": getattr(relationship, "type", None),
        "metadata": {
            "weight": getattr(relationship, "weight", 1.0),
            "attributes": getattr(relationship, "attributes", {})
        }
    }
