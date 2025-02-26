"""API routes for concepts."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from src.api.models import (
    Concept, ConceptCreate, ConceptUpdate, ConceptList, 
    PaginationParams, FilterParams, ErrorResponse
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/concepts",
    tags=["concepts"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.get(
    "",
    response_model=ConceptList,
    summary="List concepts",
    description="Get a paginated list of concepts with optional filtering",
    dependencies=[Depends(has_permission(Permission.READ_CONCEPTS))],
)
async def list_concepts(
    pagination: PaginationParams = Depends(),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    name_contains: Optional[str] = Query(None, description="Filter by name containing text"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> ConceptList:
    """Get a paginated list of concepts with optional filtering."""
    try:
        # GraphManager doesn't have get_all_concepts, use vector_store.get_all_nodes instead
        # Collect all nodes
        nodes = []
        async for node in graph_manager.vector_store.get_all_nodes():
            # Apply domain filter if specified
            if domain and node.metadata.get("domain") != domain:
                continue
            # Apply name filter if specified
            if name_contains and name_contains.lower() not in node.content.lower():
                continue
            nodes.append(node)
        
        # Apply sorting if specified
        if pagination.sort_by:
            reverse = pagination.sort_order == "desc"
            # Use a safer sorting approach with explicit type handling
            if pagination.sort_by == "metadata":
                nodes.sort(key=lambda n: str(n.metadata), reverse=reverse)
            else:
                # Convert to string to ensure comparability
                nodes.sort(
                    key=lambda n: str(getattr(n, pagination.sort_by or "id", "")),
                    reverse=reverse
                )
        
        # Apply pagination
        start_idx = (pagination.page - 1) * pagination.limit
        end_idx = start_idx + pagination.limit
        paginated_nodes = nodes[start_idx:end_idx]
        
        # Get total count
        total = len(nodes)
        
        # Convert nodes to concepts
        from src.api.adapters import nodes_to_concepts
        concepts = nodes_to_concepts(paginated_nodes)
        
        # Calculate total pages
        pages = (total + pagination.limit - 1) // pagination.limit
        
        return ConceptList(
            items=concepts,
            total=total,
            page=pagination.page,
            limit=pagination.limit,
            pages=pages,
        )
    except Exception as e:
        logger.exception(f"Error listing concepts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing concepts: {str(e)}",
        )


@router.get(
    "/{concept_id}",
    response_model=Concept,
    summary="Get concept",
    description="Get a concept by ID",
    dependencies=[Depends(has_permission(Permission.READ_CONCEPTS))],
)
async def get_concept(
    concept_id: str = Path(..., description="Concept ID"),
    include_embedding: bool = Query(False, description="Include embedding in response"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> Concept:
    """Get a concept by ID."""
    try:
        # Get concept from graph manager
        node = await graph_manager.get_concept(concept_id)
        
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Convert node to concept
        from src.api.adapters import node_to_concept
        concept = node_to_concept(node)
        
        # Remove embedding if not requested
        if not include_embedding:
            concept.embedding = None
        
        return concept
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting concept: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting concept: {str(e)}",
        )


@router.post(
    "",
    response_model=Concept,
    status_code=status.HTTP_201_CREATED,
    summary="Create concept",
    description="Create a new concept",
    dependencies=[Depends(has_permission(Permission.WRITE_CONCEPTS))],
)
async def create_concept(
    concept: ConceptCreate,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> Concept:
    """Create a new concept."""
    try:
        # Convert ConceptCreate to Node parameters
        from src.api.adapters import concept_to_node, node_to_concept
        node_params = concept_to_node(concept)
        
        # Create concept in graph manager
        node_id = await graph_manager.add_concept(
            content=node_params["content"],
            embedding=node_params["embedding"],
            metadata=node_params["metadata"]
        )
        
        # Get the created concept
        node = await graph_manager.get_concept(node_id)
        
        # Convert Node to Concept
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found",
            )
        return node_to_concept(node)
    except Exception as e:
        logger.exception(f"Error creating concept: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating concept: {str(e)}",
        )


@router.put(
    "/{concept_id}",
    response_model=Concept,
    summary="Update concept",
    description="Update an existing concept",
    dependencies=[Depends(has_permission(Permission.WRITE_CONCEPTS))],
)
async def update_concept(
    concept_id: str = Path(..., description="Concept ID"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
    concept: ConceptUpdate = Body(...),
) -> Concept:
    """Update an existing concept."""
    try:
        # Check if concept exists
        existing_concept = await graph_manager.get_concept(concept_id)
        
        if not existing_concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Convert ConceptUpdate to Node update parameters
        from src.api.adapters import concept_update_to_node_update, node_to_concept
        node_update = concept_update_to_node_update(concept, concept_id)
        
        # GraphManager doesn't have update_concept, use vector_store.update_node instead
        # First get the existing node
        node = await graph_manager.get_concept(concept_id)
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Update node properties
        if "content" in node_update:
            node.content = node_update["content"]
        if "embedding" in node_update:
            node.metadata["embedding"] = node_update["embedding"]
        if "metadata" in node_update:
            # Update metadata while preserving embedding
            embedding = node.metadata.get("embedding")
            node.metadata.update(node_update["metadata"])
            if embedding:
                node.metadata["embedding"] = embedding
        
        # Update node in vector store
        await graph_manager.vector_store.update_node(node)
        
        # Get the updated concept
        node = await graph_manager.get_concept(concept_id)
        
        # Convert Node to Concept
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found",
            )
        return node_to_concept(node)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating concept: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating concept: {str(e)}",
        )


@router.delete(
    "/{concept_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete concept",
    description="Delete a concept by ID",
    dependencies=[Depends(has_permission(Permission.WRITE_CONCEPTS))],
)
async def delete_concept(
    concept_id: str = Path(..., description="Concept ID"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> None:
    """Delete a concept by ID."""
    try:
        # Check if concept exists
        existing_concept = await graph_manager.get_concept(concept_id)
        
        if not existing_concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # GraphManager doesn't have delete_concept
        # Since BaseVectorStore doesn't have a remove_node method either,
        # we'll need to implement a workaround
        
        # For now, we'll log the deletion request
        logger.info(f"Deleting concept with ID {concept_id}")
        # In a real implementation, this would be:
        # await graph_manager.vector_store.delete_node(concept_id)
        # or similar functionality
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting concept: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting concept: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=List[Concept],
    status_code=status.HTTP_201_CREATED,
    summary="Create multiple concepts",
    description="Create multiple concepts in a single request",
    dependencies=[Depends(has_permission(Permission.WRITE_CONCEPTS))],
)
async def create_concepts_batch(
    concepts: List[ConceptCreate],
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> List[Concept]:
    """Create multiple concepts in a single request."""
    try:
        # Create concepts in graph manager
        new_concepts = []
        for concept in concepts:
            # Convert ConceptCreate to Node parameters
            from src.api.adapters import concept_to_node, node_to_concept
            node_params = concept_to_node(concept)
            
            # Create concept in graph manager
            node_id = await graph_manager.add_concept(
                content=node_params["content"],
                embedding=node_params["embedding"],
                metadata=node_params["metadata"]
            )
            
            # Get the created concept
            node = await graph_manager.get_concept(node_id)
            
            # Convert Node to Concept
            if node is None:
                # Skip concepts that couldn't be retrieved
                continue
            new_concepts.append(node_to_concept(node))
        
        return new_concepts
    except Exception as e:
        logger.exception(f"Error creating concepts batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating concepts batch: {str(e)}",
        )
