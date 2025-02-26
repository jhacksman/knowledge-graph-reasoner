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
        # Get concepts from graph manager
        nodes = await graph_manager.get_all_concepts(
            domain=domain,
            name_contains=name_contains,
            skip=(pagination.page - 1) * pagination.limit,
            limit=pagination.limit,
            sort_by=pagination.sort_by,
            sort_order=pagination.sort_order or "asc",
        )
        
        # Get total count
        total = len(await graph_manager.get_all_concepts(domain=domain, name_contains=name_contains))
        
        # Convert nodes to concepts
        from src.api.adapters import nodes_to_concepts
        concepts = nodes_to_concepts(nodes)
        
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
        
        # Update concept in graph manager
        await graph_manager.update_concept(
            concept_id=concept_id,
            content=node_update.get("content"),
            embedding=node_update.get("embedding"),
            metadata=node_update.get("metadata", {})
        )
        
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
        
        # Delete concept from graph manager
        await graph_manager.delete_concept(concept_id)
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
