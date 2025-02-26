"""API routes for concepts."""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from src.api.models import (
    Concept,
    ConceptCreate,
    ConceptUpdate,
    ConceptList,
    PaginationParams,
    ErrorResponse,
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
from src.vector_store.milvus_store import MilvusVectorStore

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/concepts",
    tags=["concepts"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.get(
    "",
    response_model=ConceptList,
    summary="List concepts",
    description="Get a paginated list of concepts with optional filtering",
)
async def list_concepts(
    pagination: PaginationParams = Depends(),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    name_contains: Optional[str] = Query(None, description="Filter by name containing string"),
    api_key: ApiKey = Depends(get_api_key),
) -> ConceptList:
    """Get a paginated list of concepts."""
    try:
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusVectorStore())
        
        # Get concepts with pagination
        concepts = await graph_manager.get_concepts(
            domain=domain,
            name_contains=name_contains,
            skip=pagination.offset,
            limit=pagination.limit,
            sort_by=pagination.sort_by,
            sort_order=pagination.sort_order,
        )
        
        # Get total count
        total = len(await graph_manager.get_concepts(domain=domain, name_contains=name_contains))
        
        # Calculate total pages
        pages = (total + pagination.limit - 1) // pagination.limit
        
        return ConceptList(
            concepts=concepts,
            total=total,
            skip=pagination.offset,
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
)
async def get_concept(
    concept_id: str,
    api_key: ApiKey = Depends(get_api_key),
) -> Concept:
    """Get a concept by ID."""
    try:
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusVectorStore())
        
        # Get concept
        concept = await graph_manager.get_concept(concept_id)
        
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
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
)
async def create_concept(
    concept: ConceptCreate,
    api_key: ApiKey = Depends(get_api_key),
    _: None = Depends(has_permission(Permission.WRITE_PERMISSION)),
) -> Concept:
    """Create a new concept."""
    try:
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusVectorStore())
        
        # Create concept
        concept_id = str(uuid.uuid4())
        created_concept = await graph_manager.add_concept(
            id=concept_id,
            name=concept.name,
            description=concept.description,
            domain=concept.domain,
            attributes=concept.attributes,
            embedding=concept.get_embedding(),
        )
        
        return created_concept
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
)
async def update_concept(
    concept_id: str,
    concept: ConceptUpdate,
    api_key: ApiKey = Depends(get_api_key),
    _: None = Depends(has_permission(Permission.WRITE_PERMISSION)),
) -> Concept:
    """Update an existing concept."""
    try:
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusVectorStore())
        
        # Check if concept exists
        existing_concept = await graph_manager.get_concept(concept_id)
        
        if not existing_concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Update concept
        updated_concept = await graph_manager.update_concept(
            id=concept_id,
            name=concept.name,
            description=concept.description,
            domain=concept.domain,
            attributes=concept.attributes,
            embedding=concept.get_embedding(),
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
)
async def delete_concept(
    concept_id: str,
    api_key: ApiKey = Depends(get_api_key),
    _: None = Depends(has_permission(Permission.WRITE_PERMISSION)),
) -> None:
    """Delete a concept by ID."""
    try:
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusVectorStore())
        
        # Check if concept exists
        existing_concept = await graph_manager.get_concept(concept_id)
        
        if not existing_concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Delete concept
        await graph_manager.delete_concept(concept_id)
        
        return None
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
    summary="Create concepts batch",
    description="Create multiple concepts in a single request",
)
async def create_concepts_batch(
    concepts: List[ConceptCreate],
    api_key: ApiKey = Depends(get_api_key),
    _: None = Depends(has_permission(Permission.WRITE_PERMISSION)),
) -> List[Concept]:
    """Create multiple concepts in a single request."""
    try:
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusVectorStore())
        
        # Create concepts
        created_concepts = []
        for concept in concepts:
            concept_id = str(uuid.uuid4())
            created_concept = await graph_manager.add_concept(
                id=concept_id,
                name=concept.name,
                description=concept.description,
                domain=concept.domain,
                attributes=concept.attributes,
                embedding=concept.get_embedding(),
            )
            created_concepts.append(created_concept)
        
        return created_concepts
    except Exception as e:
        logger.exception(f"Error creating concepts batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating concepts batch: {str(e)}",
        )
