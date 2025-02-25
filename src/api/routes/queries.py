"""API routes for queries."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, BackgroundTasks
from src.api.models import (
    Query, QueryCreate, QueryResult, QueryList, 
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
QUERIES = {}


async def execute_query(query_id: str, query_text: str, max_results: int, include_reasoning: bool):
    """Execute a query in the background."""
    try:
        # Update query status
        QUERIES[query_id]["status"] = "processing"
        
        # Execute query
        query_executor = QueryExecutor()
        result = await query_executor.execute(
            query=query_text,
            max_results=max_results,
            include_reasoning=include_reasoning,
        )
        
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
    dependencies=[Depends(has_permission(Permission.READ_QUERIES))],
)
async def list_queries(
    pagination: PaginationParams = Depends(),
    status: Optional[str] = Query(None, description="Filter by status"),
    api_key: ApiKey = Depends(get_api_key),
) -> QueryList:
    """Get a paginated list of queries."""
    try:
        # Filter queries by status if provided
        filtered_queries = [
            query for query_id, query in QUERIES.items()
            if not status or query["status"] == status
        ]
        
        # Apply pagination
        start = (pagination.page - 1) * pagination.limit
        end = start + pagination.limit
        paginated_queries = filtered_queries[start:end]
        
        # Convert to Query objects
        queries = [
            Query(
                id=query_id,
                query=query["query"],
                max_results=query["max_results"],
                include_reasoning=query["include_reasoning"],
                status=query["status"],
                result=query.get("result"),
                created_at=query["created_at"],
                completed_at=query.get("completed_at"),
            )
            for query_id, query in paginated_queries
        ]
        
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing queries: {str(e)}",
        )


@router.get(
    "/{query_id}",
    response_model=Query,
    summary="Get query",
    description="Get a query by ID",
    dependencies=[Depends(has_permission(Permission.READ_QUERIES))],
)
async def get_query(
    query_id: str = Path(..., description="Query ID"),
    api_key: ApiKey = Depends(get_api_key),
) -> Query:
    """Get a query by ID."""
    try:
        if query_id not in QUERIES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query with ID {query_id} not found",
            )
        
        query = QUERIES[query_id]
        
        return Query(
            id=query_id,
            query=query["query"],
            max_results=query["max_results"],
            include_reasoning=query["include_reasoning"],
            status=query["status"],
            result=query.get("result"),
            created_at=query["created_at"],
            completed_at=query.get("completed_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting query: {str(e)}",
        )


@router.post(
    "",
    response_model=Query,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create query",
    description="Create and execute a new query",
    dependencies=[Depends(has_permission(Permission.WRITE_QUERIES))],
)
async def create_query(
    query: QueryCreate,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
) -> Query:
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
            max_results=query.max_results,
            include_reasoning=query.include_reasoning,
        )
        
        return Query(
            id=query_id,
            query=query.query,
            max_results=query.max_results,
            include_reasoning=query.include_reasoning,
            status="pending",
            created_at=QUERIES[query_id]["created_at"],
        )
    except Exception as e:
        logger.exception(f"Error creating query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating query: {str(e)}",
        )


@router.delete(
    "/{query_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete query",
    description="Delete a query by ID",
    dependencies=[Depends(has_permission(Permission.WRITE_QUERIES))],
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
                detail=f"Query with ID {query_id} not found",
            )
        
        # Delete query
        del QUERIES[query_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting query: {str(e)}",
        )
