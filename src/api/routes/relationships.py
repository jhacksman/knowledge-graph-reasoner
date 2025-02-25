"""API routes for relationships."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from src.api.models import (
    Relationship, RelationshipCreate, RelationshipUpdate, RelationshipList, 
    PaginationParams, FilterParams, ErrorResponse
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
import logging
# Setup logging
logger = logging.getLogger(__name__)
# Create router
router = APIRouter(
    prefix="/relationships",
    tags=["relationships"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@router.get(
    "",
    response_model=RelationshipList,
    summary="List relationships",
    description="Get a paginated list of relationships with optional filtering",
    dependencies=[Depends(has_permission(Permission.READ_RELATIONSHIPS))],
)
async def list_relationships(
    pagination: PaginationParams = Depends(),
    source_id: Optional[str] = Query(None, description="Filter by source concept ID"),
    target_id: Optional[str] = Query(None, description="Filter by target concept ID"),
    type: Optional[str] = Query(None, description="Filter by relationship type"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> RelationshipList:
    """Get a paginated list of relationships with optional filtering."""
    try:
        # Get relationships from graph manager
        relationships = await graph_manager.get_relationships(
            source_id=source_id,
            target_id=target_id,
            type=type,
            skip=(pagination.page - 1) * pagination.limit,
            limit=pagination.limit,
            sort_by=pagination.sort_by,
            sort_order=pagination.sort_order,
        )
        
        # Get total count
        total = len(await graph_manager.get_all_edges(source_id=source_id, target_id=target_id, type=type))
        # Calculate total pages
        pages = (total + pagination.limit - 1) // pagination.limit
        
        return RelationshipList(
            items=relationships,
            total=total,
            page=pagination.page,
            limit=pagination.limit,
            pages=pages,
        )
    except Exception as e:
        logger.exception(f"Error listing relationships: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing relationships: {str(e)}",
        )
@router.get(
    "/{relationship_id}",
    response_model=Relationship,
    summary="Get relationship",
    description="Get a relationship by ID",
    dependencies=[Depends(has_permission(Permission.READ_RELATIONSHIPS))],
)
async def get_relationship(
    relationship_id: str = Path(..., description="Relationship ID"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> Relationship:
    """Get a relationship by ID."""
    try:
        # Get relationship from graph manager
        relationship = await graph_manager.get_edge(relationship_id)
        
        if not relationship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Relationship with ID {relationship_id} not found",
            )
        
        return relationship
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting relationship: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting relationship: {str(e)}",
        )
@router.post(
    "",
    response_model=Relationship,
    status_code=status.HTTP_201_CREATED,
    summary="Create relationship",
    description="Create a new relationship",
    dependencies=[Depends(has_permission(Permission.WRITE_RELATIONSHIPS))],
)
async def create_relationship(
    relationship: RelationshipCreate,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> Relationship:
    """Create a new relationship."""
    try:
        # Check if source and target concepts exist
        source_concept = await graph_manager.get_concept(relationship.source_id)
        if not source_concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source concept with ID {relationship.source_id} not found",
            )
        
        target_concept = await graph_manager.get_concept(relationship.target_id)
        if not target_concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target concept with ID {relationship.target_id} not found",
            )
        
        # Create relationship in graph manager
        new_relationship = await graph_manager.add_edge(
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            type=relationship.type,
            weight=relationship.weight,
            attributes=relationship.attributes,
        )
        
        return new_relationship
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating relationship: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating relationship: {str(e)}",
        )
@router.put(
    "/{relationship_id}",
    response_model=Relationship,
    summary="Update relationship",
    description="Update an existing relationship",
    dependencies=[Depends(has_permission(Permission.WRITE_RELATIONSHIPS))],
)
async def update_relationship(
    relationship_id: str = Path(..., description="Relationship ID"),
    relationship: RelationshipUpdate = None,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> Relationship:
    """Update an existing relationship."""
    try:
        # Check if relationship exists
        existing_relationship = await graph_manager.get_edge(relationship_id)
        
        if not existing_relationship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Relationship with ID {relationship_id} not found",
            )
        
        # Update relationship in graph manager
        updated_relationship = await graph_manager.update_edge(
            relationship_id=relationship_id,
            type=relationship.type,
            weight=relationship.weight,
            attributes=relationship.attributes,
        )
        
        return updated_relationship
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating relationship: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating relationship: {str(e)}",
        )
@router.delete(
    "/{relationship_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete relationship",
    description="Delete a relationship by ID",
    dependencies=[Depends(has_permission(Permission.WRITE_RELATIONSHIPS))],
)
async def delete_relationship(
    relationship_id: str = Path(..., description="Relationship ID"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> None:
    """Delete a relationship by ID."""
    try:
        # Check if relationship exists
        existing_relationship = await graph_manager.get_edge(relationship_id)
        
        if not existing_relationship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Relationship with ID {relationship_id} not found",
            )
        
        # Delete relationship from graph manager
        await graph_manager.delete_edge(relationship_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting relationship: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting relationship: {str(e)}",
        )
@router.post(
    "/batch",
    response_model=List[Relationship],
    status_code=status.HTTP_201_CREATED,
    summary="Create multiple relationships",
    description="Create multiple relationships in a single request",
    dependencies=[Depends(has_permission(Permission.WRITE_RELATIONSHIPS))],
)
async def create_relationships_batch(
    relationships: List[RelationshipCreate],
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> List[Relationship]:
    """Create multiple relationships in a single request."""
    try:
        # Create relationships in graph manager
        new_relationships = []
        for relationship in relationships:
            # Check if source and target concepts exist
            source_concept = await graph_manager.get_concept(relationship.source_id)
            if not source_concept:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source concept with ID {relationship.source_id} not found",
                )
            
            target_concept = await graph_manager.get_concept(relationship.target_id)
            if not target_concept:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Target concept with ID {relationship.target_id} not found",
                )
            
            # Create relationship
            new_relationship = await graph_manager.add_edge(
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                type=relationship.type,
                weight=relationship.weight,
                attributes=relationship.attributes,
            )
            new_relationships.append(new_relationship)
        
        return new_relationships
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating relationships batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating relationships batch: {str(e)}",
        )