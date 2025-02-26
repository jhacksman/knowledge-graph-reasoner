"""API routes for semantic search."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from src.api.models import (
    SearchRequest, SearchResponse, SearchResult,
    ErrorResponse
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
from src.reasoning.llm import VeniceLLM, VeniceLLMConfig
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic search",
    description="Perform semantic search across concepts",
    dependencies=[Depends(has_permission(Permission.READ_CONCEPTS))],
)
async def search_concepts(
    request: SearchRequest,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> SearchResponse:
    """Perform semantic search across concepts."""
    try:
        # Initialize LLM for embedding
        llm = VeniceLLM(config=VeniceLLMConfig(api_key=""))  # Empty string for type checking, real key from env vars
        
        # Generate embedding for query
        query_embedding = await llm.embed_text(request.query)
        
        # Search for concepts by embedding
        similar_nodes = await graph_manager.get_similar_concepts(
            embedding=query_embedding,
            k=request.limit,
            threshold=float(request.threshold) if request.threshold is not None else 0.7
        )
        
        # Convert nodes to concepts
        from src.api.adapters import nodes_to_concepts
        concepts = nodes_to_concepts(similar_nodes)
        
        # Create result tuples with similarity scores (using 1.0 as default since we don't have actual scores)
        concept_results = [(concept, 1.0) for concept in concepts]
        
        # Create search results
        results = []
        for concept, similarity in concept_results:
            results.append(
                SearchResult(
                    concept=concept,
                    similarity=similarity,
                )
            )
        
        return SearchResponse(
            results=results,
            query=request.query,
            total=len(results),
        )
    except Exception as e:
        logger.exception(f"Error performing search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}",
        )


@router.get(
    "/by-text/{text}",
    response_model=SearchResponse,
    summary="Search by text",
    description="Search concepts by text",
    dependencies=[Depends(has_permission(Permission.READ_CONCEPTS))],
)
async def search_by_text(
    text: str = Path(..., description="Search text"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    domains: Optional[List[str]] = Query(None, description="Domains to search in"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> SearchResponse:
    """Search concepts by text."""
    try:
        # Initialize LLM for embedding
        llm = VeniceLLM(config=VeniceLLMConfig(api_key=""))  # Empty string for type checking, real key from env vars
        
        # Generate embedding for query
        query_embedding = await llm.embed_text(text)
        
        # Search for concepts by embedding
        similar_nodes = await graph_manager.get_similar_concepts(
            embedding=query_embedding,
            k=limit,
            threshold=float(threshold) if threshold is not None else 0.7,
        )
        
        # Convert nodes to concepts
        from src.api.adapters import nodes_to_concepts
        concepts = nodes_to_concepts(similar_nodes)
        
        # Create result tuples with similarity scores (using 1.0 as default since we don't have actual scores)
        concept_results = [(concept, 1.0) for concept in concepts]
        
        # Create search results
        results = []
        for concept, similarity in concept_results:
            results.append(
                SearchResult(
                    concept=concept,
                    similarity=similarity,
                )
            )
        
        return SearchResponse(
            results=results,
            query=text,
            total=len(results),
        )
    except Exception as e:
        logger.exception(f"Error performing search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}",
        )


@router.get(
    "/by-concept/{concept_id}",
    response_model=SearchResponse,
    summary="Search by concept",
    description="Find concepts similar to a given concept",
    dependencies=[Depends(has_permission(Permission.READ_CONCEPTS))],
)
async def search_by_concept(
    concept_id: str = Path(..., description="Concept ID"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    domains: Optional[List[str]] = Query(None, description="Domains to search in"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> SearchResponse:
    """Find concepts similar to a given concept."""
    try:
        # Get concept
        node = await graph_manager.get_concept(concept_id)
        
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Convert node to concept
        from src.api.adapters import node_to_concept
        concept = node_to_concept(node)
        
        # Get concept embedding from metadata
        embedding = node.metadata.get("embedding")
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Concept with ID {concept_id} has no embedding",
            )
        
        # Search for concepts by embedding
        similar_nodes = await graph_manager.get_similar_concepts(
            embedding=embedding,
            k=limit + 1,  # Add 1 to account for the concept itself
            threshold=float(threshold) if threshold is not None else 0.7,
        )
        
        # Convert nodes to concepts
        from src.api.adapters import nodes_to_concepts
        concepts = nodes_to_concepts(similar_nodes)
        
        # Create result tuples with similarity scores (using 1.0 as default since we don't have actual scores)
        concept_results = [(concept, 1.0) for concept in concepts]
        
        # Filter out the concept itself
        concept_results = [(c, s) for c, s in concept_results if c.id != concept_id]
        
        # Limit results
        concept_results = concept_results[:limit]
        
        # Create search results
        results = []
        for concept, similarity in concept_results:
            results.append(
                SearchResult(
                    concept=concept,
                    similarity=similarity,
                )
            )
        
        return SearchResponse(
            results=results,
            query=f"Similar to concept: {concept.name}",
            total=len(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error performing search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}",
        )
