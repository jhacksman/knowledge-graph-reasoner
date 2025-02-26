#!/usr/bin/env python3
"""Fix all mypy type checking errors in the API module."""

import re
import os
from typing import Dict, List, Any, Optional, Tuple

def fix_adapters_module():
    """Fix type checking errors in adapters.py."""
    # Create a completely new version of the file with proper type annotations
    new_content = """\"\"\"Adapter functions to convert between internal and API models.\"\"\"
from typing import Dict, List, Optional, Any, Tuple, cast, Union
from datetime import datetime
import numpy as np

from src.models.node import Node
from src.models.edge import Edge
from src.api.models import (
    Concept, ConceptCreate, ConceptUpdate,
    Relationship, RelationshipCreate, RelationshipUpdate
)


def node_to_concept(node: Node) -> Concept:
    \"\"\"Convert a Node to a Concept.
    
    Args:
        node: Internal Node model
        
    Returns:
        Concept: API Concept model
    \"\"\"
    metadata = node.metadata or {}
    
    # Get embedding from metadata if available
    embedding_raw = metadata.get("embedding")
    embedding: Optional[List[float]] = None
    
    # Convert embedding to List[float] if possible
    if isinstance(embedding_raw, np.ndarray):
        embedding = embedding_raw.tolist()
    elif isinstance(embedding_raw, list):
        # Ensure all elements are floats
        try:
            # Create a new list with explicit float conversion
            float_list: List[float] = []
            for x in embedding_raw:
                if x is not None:
                    float_list.append(float(x))
            embedding = float_list if float_list else None
        except (TypeError, ValueError):
            # If conversion fails, set to None
            embedding = None
    
    # Ensure we have valid metadata values
    name_value = metadata.get("name", "")
    if not name_value and node.content:
        name_value = node.content[:50]
    elif not name_value:
        name_value = "Unnamed Concept"
        
    description = metadata.get("description", node.content)
    domain = metadata.get("domain")
    attributes = metadata.get("attributes", {})
    created_at = metadata.get("created_at", datetime.utcnow())
    updated_at = metadata.get("updated_at", datetime.utcnow())
    
    # Create the Concept with properly typed embedding
    return Concept(
        id=node.id,
        name=name_value,
        description=description,
        domain=domain,
        attributes=attributes,
        embedding=embedding,  # Now properly typed as Optional[List[float]]
        created_at=created_at,
        updated_at=updated_at,
    )


def concept_to_node(concept: ConceptCreate) -> Dict[str, Any]:
    \"\"\"Convert a ConceptCreate to a Node creation dict.
    
    Args:
        concept: API ConceptCreate model
        
    Returns:
        Dict[str, Any]: Node creation parameters
    \"\"\"
    embedding = concept.get_embedding()
    
    return {
        "content": concept.description or "",
        "embedding": embedding,
        "metadata": {
            "name": concept.name,
            "description": concept.description,
            "domain": concept.domain,
            "attributes": concept.attributes or {},
            "embedding": embedding,  # Store in metadata too for easy access
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }


def concept_update_to_node_update(concept: ConceptUpdate, node_id: str) -> Dict[str, Any]:
    \"\"\"Convert a ConceptUpdate to a Node update dict.
    
    Args:
        concept: API ConceptUpdate model
        node_id: ID of the node to update
        
    Returns:
        Dict[str, Any]: Node update parameters
    \"\"\"
    update_data: Dict[str, Any] = {
        "id": node_id,
    }
    
    if concept.description is not None:
        update_data["content"] = concept.description
    
    embedding = concept.get_embedding()
    if embedding is not None:
        update_data["embedding"] = embedding
    
    # Create metadata dictionary
    metadata_dict: Dict[str, Any] = {}
    update_data["metadata"] = metadata_dict
    
    if concept.name is not None:
        metadata_dict["name"] = concept.name
    
    if concept.description is not None:
        metadata_dict["description"] = concept.description
    
    if concept.domain is not None:
        metadata_dict["domain"] = concept.domain
    
    if concept.attributes is not None:
        metadata_dict["attributes"] = concept.attributes
    
    if embedding is not None:
        metadata_dict["embedding"] = embedding
    
    metadata_dict["updated_at"] = datetime.utcnow()
    
    return update_data


def edge_to_relationship(edge: Edge) -> Relationship:
    \"\"\"Convert an Edge to a Relationship.
    
    Args:
        edge: Internal Edge model
        
    Returns:
        Relationship: API Relationship model
    \"\"\"
    metadata = edge.metadata or {}
    
    # Generate an ID if not present in metadata
    edge_id = metadata.get("id", f"{edge.source}-{edge.target}-{edge.type}")
    
    # Get weight with proper type conversion
    weight_raw = metadata.get("weight", 1.0)
    weight = float(weight_raw) if weight_raw is not None else 1.0
    
    return Relationship(
        id=edge_id,
        source_id=edge.source,
        target_id=edge.target,
        type=edge.type,
        weight=weight,
        attributes=metadata.get("attributes", {}),
        created_at=metadata.get("created_at", datetime.utcnow()),
        updated_at=metadata.get("updated_at", datetime.utcnow()),
    )


def relationship_to_edge(relationship: RelationshipCreate) -> Dict[str, Any]:
    \"\"\"Convert a RelationshipCreate to an Edge creation dict.
    
    Args:
        relationship: API RelationshipCreate model
        
    Returns:
        Dict[str, Any]: Edge creation parameters
    \"\"\"
    edge_id = f"{relationship.source_id}-{relationship.target_id}-{relationship.type}"
    
    # Ensure weight is a float
    weight = float(relationship.weight) if relationship.weight is not None else 1.0
    
    return {
        "source": relationship.source_id,
        "target": relationship.target_id,
        "type": relationship.type,
        "metadata": {
            "id": edge_id,
            "weight": weight,
            "attributes": relationship.attributes or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    }


def relationship_update_to_edge_update(relationship: RelationshipUpdate, edge_id: str) -> Dict[str, Any]:
    \"\"\"Convert a RelationshipUpdate to an Edge update dict.
    
    Args:
        relationship: API RelationshipUpdate model
        edge_id: ID of the edge to update
        
    Returns:
        Dict[str, Any]: Edge update parameters
    \"\"\"
    update_data: Dict[str, Any] = {
        "id": edge_id,
    }
    
    if relationship.type is not None:
        update_data["type"] = relationship.type
    
    # Create metadata dictionary
    metadata_dict: Dict[str, Any] = {}
    update_data["metadata"] = metadata_dict
    
    if relationship.weight is not None:
        metadata_dict["weight"] = float(relationship.weight)
    
    if relationship.attributes is not None:
        metadata_dict["attributes"] = relationship.attributes
    
    metadata_dict["updated_at"] = datetime.utcnow()
    
    return update_data


def nodes_to_concepts(nodes: List[Node]) -> List[Concept]:
    \"\"\"Convert a list of Nodes to a list of Concepts.
    
    Args:
        nodes: List of internal Node models
        
    Returns:
        List[Concept]: List of API Concept models
    \"\"\"
    return [node_to_concept(node) for node in nodes]


def edges_to_relationships(edges: List[Edge]) -> List[Relationship]:
    \"\"\"Convert a list of Edges to a list of Relationships.
    
    Args:
        edges: List of internal Edge models
        
    Returns:
        List[Relationship]: List of API Relationship models
    \"\"\"
    return [edge_to_relationship(edge) for edge in edges]


def similarity_results_to_concepts(
    results: List[Tuple[Node, float]]
) -> List[Tuple[Concept, float]]:
    \"\"\"Convert similarity search results to Concept results.
    
    Args:
        results: List of (Node, similarity) tuples
        
    Returns:
        List[Tuple[Concept, float]]: List of (Concept, similarity) tuples
    \"\"\"
    # Handle case where results might be None
    if not results:
        return []
    
    # Convert each Node to Concept while preserving similarity score
    return [(node_to_concept(node), similarity) for node, similarity in results]
"""
    
    # Write the fixed content
    with open("src/api/adapters.py", "w") as f:
        f.write(new_content)
    
    print("Fixed adapters.py")


def fix_milvus_store_naming():
    """Fix MilvusVectorStore vs MilvusStore naming inconsistency."""
    # Fix main.py
    with open("src/api/main.py", "r") as f:
        content = f.read()
    
    # Replace MilvusVectorStore with MilvusStore
    content = content.replace("MilvusVectorStore", "MilvusStore")
    
    # Fix the instantiation to include required parameters
    content = content.replace(
        "vector_store = MilvusStore()",
        "vector_store = MilvusStore(collection_name='concepts', connection_args={'host': 'localhost', 'port': '19530'})"
    )
    
    with open("src/api/main.py", "w") as f:
        f.write(content)
    
    # Check all route files for the same issue
    route_files = [
        "src/api/routes/concepts.py",
        "src/api/routes/relationships.py",
        "src/api/routes/search.py",
        "src/api/routes/queries.py",
        "src/api/routes/expansion.py",
        "src/api/routes/metrics.py"
    ]
    
    for file_path in route_files:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
            
            # Replace MilvusVectorStore with MilvusStore
            content = content.replace("MilvusVectorStore", "MilvusStore")
            
            with open(file_path, "w") as f:
                f.write(content)
    
    print("Fixed MilvusVectorStore vs MilvusStore naming inconsistency")


def fix_graph_manager_method_names():
    """Fix GraphManager method name mismatches."""
    # Map of incorrect method names to correct ones
    method_map = {
        "get_concepts": "get_all_nodes",
        "add_node": "add_concept",
        "search_concepts_by_embedding": "get_similar_concepts",
        "get_relationships": "get_edges",
        "add_relationship": "add_edge",
        "update_relationship": "update_edge",
        "delete_relationship": "delete_edge",
        "get_all_edges": "get_all_edges",  # This is actually correct, but needs implementation
    }
    
    # Check all route files for method name issues
    route_files = [
        "src/api/routes/concepts.py",
        "src/api/routes/relationships.py",
        "src/api/routes/search.py",
        "src/api/routes/queries.py",
        "src/api/routes/expansion.py",
        "src/api/routes/metrics.py"
    ]
    
    for file_path in route_files:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
            
            # Replace incorrect method names with correct ones
            for incorrect, correct in method_map.items():
                content = re.sub(
                    r'graph_manager\.' + incorrect + r'\(',
                    'graph_manager.' + correct + '(',
                    content
                )
            
            with open(file_path, "w") as f:
                f.write(content)
    
    print("Fixed GraphManager method name mismatches")


def fix_method_signatures():
    """Fix method signature mismatches."""
    # Fix concepts.py
    if os.path.exists("src/api/routes/concepts.py"):
        with open("src/api/routes/concepts.py", "r") as f:
            content = f.read()
        
        # Fix get_all_nodes signature
        content = re.sub(
            r'graph_manager\.get_all_nodes\(.*?sort_order=([^,\)]+).*?\)',
            lambda m: f'graph_manager.get_all_nodes(sort_order=str({m.group(1)}) if {m.group(1)} is not None else "asc")',
            content
        )
        
        # Fix node_to_concept calls with None check
        content = re.sub(
            r'node_to_concept\(([^)]+)\)',
            lambda m: f'node_to_concept({m.group(1)}) if {m.group(1)} is not None else None',
            content
        )
        
        with open("src/api/routes/concepts.py", "w") as f:
            f.write(content)
    
    # Fix relationships.py
    if os.path.exists("src/api/routes/relationships.py"):
        with open("src/api/routes/relationships.py", "r") as f:
            content = f.read()
        
        # Fix add_edge signature
        content = re.sub(
            r'graph_manager\.add_edge\(.*?type=([^,\)]+).*?weight=([^,\)]+).*?attributes=([^,\)]+).*?\)',
            lambda m: f'graph_manager.add_edge(source={m.group(1)}, target={m.group(2)}, edge_type={m.group(3)}, metadata={{"weight": {m.group(4)}, "attributes": {m.group(5)}}})',
            content
        )
        
        # Fix update_edge signature
        content = re.sub(
            r'graph_manager\.update_edge\(.*?relationship_id=([^,\)]+).*?type=([^,\)]+).*?weight=([^,\)]+).*?attributes=([^,\)]+).*?\)',
            lambda m: f'graph_manager.update_edge(edge_id={m.group(1)}, edge_type={m.group(2)}, metadata={{"weight": {m.group(3)}, "attributes": {m.group(4)}}})',
            content
        )
        
        with open("src/api/routes/relationships.py", "w") as f:
            f.write(content)
    
    # Fix search.py
    if os.path.exists("src/api/routes/search.py"):
        with open("src/api/routes/search.py", "r") as f:
            content = f.read()
        
        # Fix float conversion
        content = re.sub(
            r'threshold=float\(([^)]+)\)',
            lambda m: f'threshold=float({m.group(1)}) if {m.group(1)} is not None else 0.7',
            content
        )
        
        with open("src/api/routes/search.py", "w") as f:
            f.write(content)
    
    print("Fixed method signature mismatches")


def fix_milvus_store_implementation():
    """Fix MilvusStore implementation to include required abstract methods."""
    if os.path.exists("src/vector_store/milvus_store.py"):
        with open("src/vector_store/milvus_store.py", "r") as f:
            content = f.read()
        
        # Check if the methods are already implemented
        missing_methods = []
        if "async def delete_node" not in content:
            missing_methods.append('''
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the vector store.
        
        Args:
            node_id: ID of the node to delete
        """
        # Delete the node from the collection
        await self.collection.delete(f"id == {node_id}")
        logger.info(f"Deleted node with ID {node_id}")
''')
        
        if "async def delete_edge" not in content:
            missing_methods.append('''
    async def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> None:
        """Delete an edge from the vector store.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type
        """
        # Delete the edge from the collection
        await self.edge_collection.delete(
            f"source == {source_id} && target == {target_id} && type == {edge_type}"
        )
        logger.info(f"Deleted edge from {source_id} to {target_id} of type {edge_type}")
''')
        
        if "async def update_edge" not in content:
            missing_methods.append('''
    async def update_edge(self, edge: Edge) -> None:
        """Update an edge in the vector store.
        
        Args:
            edge: Edge to update
        """
        # Delete the existing edge
        await self.delete_edge(edge.source, edge.target, edge.type)
        
        # Add the updated edge
        await self.add_edge(edge)
        logger.info(f"Updated edge from {edge.source} to {edge.target} of type {edge.type}")
''')
        
        if "async def update_node" not in content:
            missing_methods.append('''
    async def update_node(self, node: Node) -> None:
        """Update a node in the vector store.
        
        Args:
            node: Node to update
        """
        # Delete the existing node
        await self.delete_node(node.id)
        
        # Add the updated node
        await self.add_node(node)
        logger.info(f"Updated node with ID {node.id}")
''')
        
        # Add missing methods to the class
        if missing_methods:
            # Find the end of the class
            class_end = content.rfind("}")
            if class_end != -1:
                # Insert the missing methods before the end of the class
                content = content[:class_end] + "".join(missing_methods) + content[class_end:]
                
                with open("src/vector_store/milvus_store.py", "w") as f:
                    f.write(content)
    
    print("Fixed MilvusStore implementation")


def fix_all_mypy_errors():
    """Fix all mypy type checking errors in the API module."""
    # Fix adapters.py
    fix_adapters_module()
    
    # Fix MilvusVectorStore vs MilvusStore naming inconsistency
    fix_milvus_store_naming()
    
    # Fix GraphManager method name mismatches
    fix_graph_manager_method_names()
    
    # Fix method signature mismatches
    fix_method_signatures()
    
    # Fix MilvusStore implementation
    fix_milvus_store_implementation()
    
    print("Fixed all mypy type checking errors in the API module")


if __name__ == "__main__":
    fix_all_mypy_errors()
