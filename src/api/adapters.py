"""Adapter functions for converting between API models and internal models."""
from typing import List, Dict, Any, Optional, Union
import uuid
from datetime import datetime
import numpy as np

from src.models.node import Node
from src.models.edge import Edge
from src.api.models import Concept, ConceptCreate, ConceptUpdate, Relationship, RelationshipCreate, RelationshipUpdate


def node_to_concept(node: Node) -> Concept:
    """Convert a Node to a Concept.
    
    Args:
        node: A Node object from the internal model
        
    Returns:
        Concept: A Concept object for the API model
    """
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
    """Convert a list of Nodes to a list of Concepts.
    
    Args:
        nodes: A list of Node objects from the internal model
        
    Returns:
        List[Concept]: A list of Concept objects for the API model
    """
    return [node_to_concept(node) for node in nodes]


def concept_to_node(concept: ConceptCreate) -> Dict[str, Any]:
    """Convert a ConceptCreate to Node parameters.
    
    Args:
        concept: A ConceptCreate object from the API model
        
    Returns:
        Dict[str, Any]: Parameters for creating a Node
    """
    # Prepare metadata
    metadata = concept.metadata or {}
    
    # Return node parameters
    return {
        "content": concept.content,
        "embedding": concept.embedding,
        "metadata": metadata
    }


def concept_update_to_node_update(concept: ConceptUpdate, concept_id: str) -> Dict[str, Any]:
    """Convert a ConceptUpdate to Node update parameters.
    
    Args:
        concept: A ConceptUpdate object from the API model
        concept_id: The ID of the concept to update
        
    Returns:
        Dict[str, Any]: Parameters for updating a Node
    """
    # Prepare update parameters
    update = {}
    
    if concept.content is not None:
        update["content"] = concept.content
    
    if concept.embedding is not None:
        update["embedding"] = concept.embedding
    
    if concept.metadata is not None:
        update["metadata"] = concept.metadata
    
    return update


def edge_to_relationship(edge: Edge) -> Relationship:
    """Convert an Edge to a Relationship.
    
    Args:
        edge: An Edge object from the internal model
        
    Returns:
        Relationship: A Relationship object for the API model
    """
    # Extract ID from edge or generate a new one
    edge_id = getattr(edge, "id", str(uuid.uuid4()))
    
    # Extract weight and attributes from metadata
    weight = edge.metadata.get("weight", 1.0) if edge.metadata else 1.0
    attributes = edge.metadata.get("attributes", {}) if edge.metadata else {}
    
    # Create relationship from edge
    return Relationship(
        id=edge_id,
        source_id=edge.source,
        target_id=edge.target,
        type=edge.type,
        weight=weight,
        attributes=attributes,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


def edges_to_relationships(edges: List[Edge]) -> List[Relationship]:
    """Convert a list of Edges to a list of Relationships.
    
    Args:
        edges: A list of Edge objects from the internal model
        
    Returns:
        List[Relationship]: A list of Relationship objects for the API model
    """
    return [edge_to_relationship(edge) for edge in edges]


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
