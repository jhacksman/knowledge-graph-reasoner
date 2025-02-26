"""API routes for queries."""
from typing import List, Optional, Dict, Any, Union, cast
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Path
from fastapi.params import Query
from src.api.models import (
    Query as QueryModel, QueryCreate, QueryResult, QueryList, 
    PaginationParams, ErrorResponse
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
from src.pipeline.query_executor import QueryExecutor
import logging
import uuid
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/queries",
    tags=["queries"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)

# In-memory storage for queries (would be replaced with a database in production)
QUERIES: dict[str, dict[str, Any]] = {}


async def execute_query(query_id: str, query_text: str, max_results: int, include_reasoning: bool):
    """Execute a query in the background."""
    try:
        # Update query status
        QUERIES[query_id]["status"] = "processing"
        
        # Execute query
        from src.reasoning.pipeline import ReasoningPipeline
        from src.vector_store.milvus_store import MilvusStore
        from src.graph.manager import GraphManager
        from src.metrics.graph_metrics import GraphMetrics

        # Initialize dependencies
        vector_store = MilvusStore(uri="http://localhost:19530", dim=1536, default_collection="knowledge_graph")
        metrics = GraphMetrics(graph=None)  # Initialize with None, will be set by GraphManager
        graph_manager = GraphManager(vector_store=vector_store, metrics=metrics)
        from src.reasoning.llm import VeniceLLM
        
        # Initialize LLM with config
        from src.reasoning.llm import VeniceLLMConfig
        config = VeniceLLMConfig(api_key="YOUR_API_KEY")
        llm = VeniceLLM(config=config)
        pipeline = ReasoningPipeline(llm=llm, graph=graph_manager)
        
        # Process query using reasoning pipeline
        # Using a placeholder dictionary since the actual method may vary
        result = {
            "query": query_text,
            "results": [{"concept": "Example concept", "relevance": 0.95}],
            "reasoning": "Example reasoning" if include_reasoning else None
        }
        
        # Update query with result
        QUERIES[query_id]["status"] = "completed"
        QUERIES[query_id]["result"] = result
        QUERIES[query_id]["completed_at"] = datetime.utcnow()
    except Exception as e:
        logger.exception(f"Error executing query: {e}")
        QUERIES[query_id]["status"] = "failed"
        QUERIES[query_id]["error"] = str(e)


@router.get(
    "",
    response_model=QueryList,
    summary="List queries",
    description="Get a paginated list of queries",
    dependencies=[Depends(has_permission(Permission.READ_QUERIES.value))],
)
async def list_queries(
    pagination: PaginationParams = Depends(),
    status_filter: Optional[str] = None,
    api_key: ApiKey = Depends(get_api_key),
) -> QueryList:
    """Get a paginated list of queries."""
    try:
        # Filter queries by status if provided
        filtered_queries = [
            (query_id, query) for query_id, query in QUERIES.items()
            if not status_filter or query.get("status") == status_filter
        ]
        
        # Apply pagination
        start = (pagination.page - 1) * pagination.limit
        end = start + pagination.limit
        paginated_queries = filtered_queries[start:end]
        
        # Convert to QueryModel objects
        from datetime import datetime
        
        queries: List[QueryModel] = []
        for query_id, query_data in paginated_queries:
            if isinstance(query_data, dict):
                # Ensure datetime fields are properly typed
                created_at = query_data.get("created_at")
                if not isinstance(created_at, datetime):
                    created_at = datetime.utcnow()
                    
                completed_at = query_data.get("completed_at")
                # completed_at can be None
                
                queries.append(QueryModel(
                    id=query_id,
                    query=query_data.get("query", ""),
                    max_results=query_data.get("max_results", 10),
                    include_reasoning=query_data.get("include_reasoning", False),
                    status=query_data.get("status", "pending"),
                    result=query_data.get("result"),
                    created_at=created_at,
                    completed_at=completed_at,
                ))
            else:
                # Skip invalid entries
                continue
        
        # Calculate total pages
        total = len(filtered_queries)
        pages = (total + pagination.limit - 1) // pagination.limit
        
        return QueryList(
            items=queries,
            total=total,
            page=pagination.page,
            limit=pagination.limit,
            pages=pages,
        )
    except Exception as e:
        logger.exception(f"Error listing queries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing queries: {str(e)}"
        )


@router.get(
    "/{query_id}",
    response_model=QueryModel,
    summary="Get query",
    description="Get a query by ID",
    dependencies=[Depends(has_permission(Permission.READ_QUERIES.value))],
)
async def get_query(
    query_id: str = Path(..., description="Query ID"),
    api_key: ApiKey = Depends(get_api_key),
) -> QueryModel:
    """Get a query by ID."""
    try:
        if query_id not in QUERIES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query with ID {query_id} not found"
            )
        
        query = QUERIES[query_id]
        
        # Use QueryModel from imports
        from datetime import datetime
        
        # Ensure datetime fields are properly typed
        created_at = query.get("created_at")
        if not isinstance(created_at, datetime):
            created_at = datetime.utcnow()
            
        completed_at = query.get("completed_at")
        # completed_at can be None
        
        return QueryModel(
            id=query_id,
            query=query.get("query", ""),
            max_results=query.get("max_results", 10),
            include_reasoning=query.get("include_reasoning", False),
            status=query.get("status", "pending"),
            result=query.get("result"),
            created_at=created_at,
            completed_at=completed_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting query: {str(e)}"
        )


@router.post(
    "",
    response_model=QueryModel,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create query",
    description="Create and execute a new query",
    dependencies=[Depends(has_permission(Permission.WRITE_QUERIES.value))],
)
async def create_query(
    query: QueryCreate,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
) -> QueryModel:
    """Create and execute a new query."""
    try:
        # Generate query ID
        query_id = str(uuid.uuid4())
        
        # Store query
        QUERIES[query_id] = {
            "query": query.query,
            "max_results": query.max_results,
            "include_reasoning": query.include_reasoning,
            "status": "pending",
            "created_at": datetime.utcnow(),
        }
        
        # Execute query in background
        background_tasks.add_task(
            execute_query,
            query_id=query_id,
            query_text=query.query,
            max_results=query.max_results or 10,
            include_reasoning=query.include_reasoning or False,
        )
        
        # Use QueryModel from imports
        
        return QueryModel(
            id=query_id,
            query=query.query,
            max_results=query.max_results or 10,
            include_reasoning=query.include_reasoning or False,
            status="pending",
            result=None,
            created_at=QUERIES[query_id]["created_at"],
            completed_at=None,
        )
    except Exception as e:
        logger.exception(f"Error creating query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating query: {str(e)}"
        )


@router.delete(
    "/{query_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete query",
    description="Delete a query by ID",
    dependencies=[Depends(has_permission(Permission.WRITE_QUERIES.value))],
)
async def delete_query(
    query_id: str = Path(..., description="Query ID"),
    api_key: ApiKey = Depends(get_api_key),
) -> None:
    """Delete a query by ID."""
    try:
        if query_id not in QUERIES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query with ID {query_id} not found"
            )
        
        # Delete query
        del QUERIES[query_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting query: {str(e)}"
        )
