"""API routes for search."""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from src.api.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    ErrorResponse,
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
from src.reasoning.llm import VeniceLLM
from src.reasoning.config import VeniceLLMConfig
from src.vector_store.milvus_store import MilvusStore

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search concepts",
    description="Perform semantic search across concepts",
)
async def search_concepts(
    request: SearchRequest,
    api_key: ApiKey = Depends(get_api_key),
) -> SearchResponse:
    """Perform semantic search across concepts."""
    try:
        # Initialize LLM for embedding
        llm = VeniceLLM(config=VeniceLLMConfig())
        
        # Generate embedding for query
        query_embedding = await llm.embed_text(request.query)
        
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusStore())
        
        # Search for concepts by embedding
        concepts = await graph_manager.get_similar_concepts(query_embedding, threshold=request.threshold)
        
        # Create search results
        results = []
        for concept, similarity in concepts:
            results.append(
                SearchResult(
                    concept=concept,
                    similarity=similarity,
                )
            )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
        )
    except Exception as e:
        logger.exception(f"Error searching concepts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching concepts: {str(e)}",
        )


@router.get(
    "/by-text/{text}",
    response_model=SearchResponse,
    summary="Search by text",
    description="Search concepts by text",
)
async def search_by_text(
    text: str,
    limit: int = Query(10, description="Maximum number of results to return"),
    threshold: float = Query(0.7, description="Minimum similarity threshold"),
    domains: Optional[List[str]] = Query(None, description="Filter by domains"),
    api_key: ApiKey = Depends(get_api_key),
) -> SearchResponse:
    """Search concepts by text."""
    try:
        # Initialize LLM for embedding
        llm = VeniceLLM(config=VeniceLLMConfig())
        
        # Generate embedding for query
        query_embedding = await llm.embed_text(text)
        
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusStore())
        
        # Search for concepts by embedding
        concepts = await graph_manager.get_similar_concepts(query_embedding, threshold=threshold)
        
        # Create search results
        results = []
        for concept, similarity in concepts:
            results.append(
                SearchResult(
                    concept=concept,
                    similarity=similarity,
                )
            )
        
        return SearchResponse(
            query=text,
            results=results,
            total=len(results),
        )
    except Exception as e:
        logger.exception(f"Error searching concepts by text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching concepts by text: {str(e)}",
        )


@router.get(
    "/by-concept/{concept_id}",
    response_model=SearchResponse,
    summary="Search by concept",
    description="Search concepts similar to a given concept",
)
async def search_by_concept(
    concept_id: str,
    limit: int = Query(10, description="Maximum number of results to return"),
    threshold: float = Query(0.7, description="Minimum similarity threshold"),
    domains: Optional[List[str]] = Query(None, description="Filter by domains"),
    api_key: ApiKey = Depends(get_api_key),
) -> SearchResponse:
    """Search concepts similar to a given concept."""
    try:
        # Get graph manager from request state
        graph_manager = GraphManager(vector_store=MilvusStore())
        
        # Get concept
        concept = await graph_manager.get_concept(concept_id)
        
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found",
            )
        
        # Search for concepts by embedding
        concepts = await graph_manager.get_similar_concepts(concept.embedding, threshold=threshold)
        
        # Create search results
        results = []
        for concept, similarity in concepts:
            results.append(
                SearchResult(
                    concept=concept,
                    similarity=similarity,
                )
            )
        
        return SearchResponse(
            query=f"Similar to concept: {concept.name}",
            results=results,
            total=len(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error searching concepts by concept: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching concepts by concept: {str(e)}",
        )
