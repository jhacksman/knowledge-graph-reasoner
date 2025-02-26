from typing import Any
"""Adapter functions to convert between internal and API models."""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from src.models.node import Node
from src.models.edge import Edge
from src.api.models import Concept, Relationship


def node_to_concept(*args, **kwargs) -> Node:
    """Convert a Node object to a Concept object."""
    metadata = node.metadata or {}
    
    return Concept(
        id=node.id,
        name=metadata.get("name", node.content[:50]),  # Use first 50 chars of content as name if not provided
        description=metadata.get("description", node.content),
        domain=metadata.get("domain"),
        attributes=metadata.get("attributes", {}),
        embedding=metadata.get("embedding"),
        created_at=metadata.get("created_at", datetime.utcnow()),
        updated_at=metadata.get("updated_at", datetime.utcnow()),
    )


def edge_to_relationship(*args, **kwargs) -> Edge:
    """Convert an Edge object to a Relationship object."""
    metadata = edge.metadata or {}
    
    return Relationship(
        id=metadata.get("id", f"{edge.source}_{edge.target}_{edge.type}"),
        source_id=edge.source,
        target_id=edge.target,
        type=edge.type,
        weight=metadata.get("weight", 1.0),
        attributes=metadata.get("attributes", {}),
        created_at=metadata.get("created_at", datetime.utcnow()),
        updated_at=metadata.get("updated_at", datetime.utcnow()),
    )


def concept_to_node(concept: Concept) -> Node:
    """Convert a Concept object to a Node object."""
    return Node(
        id=concept.id,
        content=concept.description or concept.name,
        metadata={
            "name": concept.name,
            "description": concept.description,
            "domain": concept.domain,
            "attributes": concept.attributes,
            "embedding": concept.embedding,
            "created_at": concept.created_at,
            "updated_at": concept.updated_at,
        }
    )


def relationship_to_edge(relationship: Relationship) -> Edge:
    """Convert a Relationship object to an Edge object."""
    return Edge(
        source=relationship.source_id,
        target=relationship.target_id,
        type=relationship.type,
        metadata={
            "id": relationship.id,
            "weight": relationship.weight,
            "attributes": relationship.attributes,
            "created_at": relationship.created_at,
            "updated_at": relationship.updated_at,
        }
    )


def nodes_to_concepts(nodes: List[Node]) -> List[Concept]:
    """Convert a list of Node objects to a list of Concept objects."""
    return [node_to_concept(node) for node in nodes]


def edges_to_relationships(edges: List[Edge]) -> List[Relationship]:
    """Convert a list of Edge objects to a list of Relationship objects."""
    return [edge_to_relationship(edge) for edge in edges]


def concepts_to_nodes(concepts: List[Concept]) -> List[Node]:
    """Convert a list of Concept objects to a list of Node objects."""
    return [concept_to_node(concept) for concept in concepts]


def relationships_to_edges(relationships: List[Relationship]) -> List[Edge]:
    """Convert a list of Relationship objects to a list of Edge objects."""
    return [relationship_to_edge(relationship) for relationship in relationships]


def node_similarity_to_concept_similarity(node_similarities: List[Tuple[Node, float]]) -> List[Tuple[Concept, float]]:
    """Convert a list of (Node, similarity) tuples to a list of (Concept, similarity) tuples."""
    return [(node_to_concept(node), similarity) for node, similarity in node_similarities]
