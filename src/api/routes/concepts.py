"""API routes for concepts."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
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
        concepts = await graph_manager.get_all_nodes(
            domain=domain,
            name_contains=name_contains,
            skip=(pagination.page - 1) * pagination.limit,
            limit=pagination.limit,
            sort_by=pagination.sort_by,
            sort_order=pagination.sort_order,
        )
        
        # Get total count
        total = len(await graph_manager.get_all_nodes(domain=domain, name_contains=name_contains))
        )
        
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
        concept = await graph_manager.get_concept(concept_id)
        
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Remove embedding if not requested
        if not include_embedding and hasattr(concept, "embedding"):
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
        # Create concept in graph manager
        new_concept = await graph_manager.add_node(
            name=concept.name,
            description=concept.description,
            domain=concept.domain,
            attributes=concept.attributes,
        )
        
        return new_concept
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
    concept: ConceptUpdate = None,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
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
        
        # Update concept in graph manager
        updated_concept = await graph_manager.update_node(
            concept_id=concept_id,
            name=concept.name,
            description=concept.description,
            domain=concept.domain,
            attributes=concept.attributes,
        )
        
        return updated_concept
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
        await graph_manager.delete_node(concept_id)
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
            new_concept = await graph_manager.add_node(
                name=concept.name,
                description=concept.description,
                domain=concept.domain,
                attributes=concept.attributes,
            )
            new_concepts.append(new_concept)
        
        return new_concepts
    except Exception as e:
        logger.exception(f"Error creating concepts batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating concepts batch: {str(e)}",
        )
