"""API routes for knowledge graph expansion."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
from datetime import datetime
import uuid

from src.api.models import (
    ExpansionConfig, ExpansionStatus, ExpansionEvent,
    ErrorResponse, PaginationParams
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
from src.reasoning.pipeline import ReasoningPipeline

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/expansion",
    tags=["expansion"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)

# In-memory storage for expansion processes (would be replaced with a database in production)
EXPANSIONS = {}
EXPANSION_EVENTS = {}


async def run_expansion(
    expansion_id: str,
    config: ExpansionConfig,
    graph_manager: GraphManager,
):
    """Run a knowledge graph expansion process in the background."""
    try:
        # Update expansion status
        EXPANSIONS[expansion_id]["status"] = "running"
        EXPANSIONS[expansion_id]["updated_at"] = datetime.utcnow()
        
        # Initialize reasoning pipeline
        pipeline = ReasoningPipeline(graph_manager=graph_manager)
        
        # Get seed concepts
        seed_concepts = []
        for concept_id in config.seed_concepts:
            concept = await graph_manager.get_concept(concept_id)
            if concept:
                seed_concepts.append(concept)
        
        if not seed_concepts:
            raise ValueError("No valid seed concepts found")
        
        # Configure pipeline
        pipeline_config = {
            "max_iterations": config.max_iterations,
            "expansion_breadth": config.expansion_breadth,
            "domains": config.domains,
            "checkpoint_interval": config.checkpoint_interval,
        }
        
        # Register event handler
        async def event_handler(event_type, data):
            event = ExpansionEvent(
                expansion_id=expansion_id,
                event_type=event_type,
                data=data,
                timestamp=datetime.utcnow(),
            )
            if expansion_id not in EXPANSION_EVENTS:
                EXPANSION_EVENTS[expansion_id] = []
            EXPANSION_EVENTS[expansion_id].append(event.dict())
            
            # Update expansion status
            if event_type == "iteration_complete":
                EXPANSIONS[expansion_id]["current_iteration"] = data.get("iteration", 0)
                EXPANSIONS[expansion_id]["updated_at"] = datetime.utcnow()
            elif event_type == "expansion_complete":
                EXPANSIONS[expansion_id]["status"] = "completed"
                EXPANSIONS[expansion_id]["updated_at"] = datetime.utcnow()
        
        # Set event handler
        pipeline.set_event_handler(event_handler)
        
        # Run expansion
        await pipeline.expand_knowledge_graph(
            seed_concepts=seed_concepts,
            config=pipeline_config,
        )
        
        # Update expansion status
        total_concepts = await graph_manager.count_concepts()
        total_relationships = await graph_manager.count_relationships()
        
        EXPANSIONS[expansion_id].update({
            "status": "completed",
            "total_concepts": total_concepts,
            "total_relationships": total_relationships,
            "updated_at": datetime.utcnow(),
        })
        
    except Exception as e:
        logger.exception(f"Error running expansion: {e}")
        EXPANSIONS[expansion_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow(),
        })


@router.post(
    "",
    response_model=ExpansionStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start expansion",
    description="Start a knowledge graph expansion process",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def start_expansion(
    config: ExpansionConfig,
    background_tasks: BackgroundTasks,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> ExpansionStatus:
    """Start a knowledge graph expansion process."""
    try:
        # Validate seed concepts
        for concept_id in config.seed_concepts:
            concept = await graph_manager.get_concept(concept_id)
            if not concept:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Seed concept with ID {concept_id} not found",
                )
        
        # Generate expansion ID
        expansion_id = str(uuid.uuid4())
        
        # Initialize expansion status
        EXPANSIONS[expansion_id] = {
            "id": expansion_id,
            "config": config.dict(),
            "current_iteration": 0,
            "total_concepts": await graph_manager.count_concepts(),
            "total_relationships": await graph_manager.count_relationships(),
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        
        # Initialize events list
        EXPANSION_EVENTS[expansion_id] = []
        
        # Start expansion in background
        background_tasks.add_task(
            run_expansion,
            expansion_id=expansion_id,
            config=config,
            graph_manager=graph_manager,
        )
        
        return ExpansionStatus(
            id=expansion_id,
            config=config,
            current_iteration=0,
            total_concepts=EXPANSIONS[expansion_id]["total_concepts"],
            total_relationships=EXPANSIONS[expansion_id]["total_relationships"],
            status="pending",
            created_at=EXPANSIONS[expansion_id]["created_at"],
            updated_at=EXPANSIONS[expansion_id]["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error starting expansion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting expansion: {str(e)}",
        )


@router.get(
    "",
    response_model=List[ExpansionStatus],
    summary="List expansions",
    description="Get a list of all expansion processes",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def list_expansions(
    status: Optional[str] = Query(None, description="Filter by status"),
    api_key: ApiKey = Depends(get_api_key),
) -> List[ExpansionStatus]:
    """Get a list of all expansion processes."""
    try:
        # Filter expansions by status if provided
        filtered_expansions = [
            ExpansionStatus(
                id=exp_id,
                config=ExpansionConfig(**exp["config"]),
                current_iteration=exp["current_iteration"],
                total_concepts=exp["total_concepts"],
                total_relationships=exp["total_relationships"],
                status=exp["status"],
                created_at=exp["created_at"],
                updated_at=exp["updated_at"],
            )
            for exp_id, exp in EXPANSIONS.items()
            if not status or exp["status"] == status
        ]
        
        return filtered_expansions
    except Exception as e:
        logger.exception(f"Error listing expansions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing expansions: {str(e)}",
        )


@router.get(
    "/{expansion_id}",
    response_model=ExpansionStatus,
    summary="Get expansion",
    description="Get the status of an expansion process",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def get_expansion(
    expansion_id: str = Path(..., description="Expansion ID"),
    api_key: ApiKey = Depends(get_api_key),
) -> ExpansionStatus:
    """Get the status of an expansion process."""
    try:
        if expansion_id not in EXPANSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expansion with ID {expansion_id} not found",
            )
        
        exp = EXPANSIONS[expansion_id]
        
        return ExpansionStatus(
            id=expansion_id,
            config=ExpansionConfig(**exp["config"]),
            current_iteration=exp["current_iteration"],
            total_concepts=exp["total_concepts"],
            total_relationships=exp["total_relationships"],
            status=exp["status"],
            created_at=exp["created_at"],
            updated_at=exp["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting expansion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting expansion: {str(e)}",
        )


@router.post(
    "/{expansion_id}/pause",
    response_model=ExpansionStatus,
    summary="Pause expansion",
    description="Pause an ongoing expansion process",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def pause_expansion(
    expansion_id: str = Path(..., description="Expansion ID"),
    api_key: ApiKey = Depends(get_api_key),
) -> ExpansionStatus:
    """Pause an ongoing expansion process."""
    try:
        if expansion_id not in EXPANSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expansion with ID {expansion_id} not found",
            )
        
        exp = EXPANSIONS[expansion_id]
        
        if exp["status"] != "running":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expansion is not running (current status: {exp['status']})",
            )
        
        # Update status
        exp["status"] = "paused"
        exp["updated_at"] = datetime.utcnow()
        
        return ExpansionStatus(
            id=expansion_id,
            config=ExpansionConfig(**exp["config"]),
            current_iteration=exp["current_iteration"],
            total_concepts=exp["total_concepts"],
            total_relationships=exp["total_relationships"],
            status=exp["status"],
            created_at=exp["created_at"],
            updated_at=exp["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error pausing expansion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error pausing expansion: {str(e)}",
        )


@router.post(
    "/{expansion_id}/resume",
    response_model=ExpansionStatus,
    summary="Resume expansion",
    description="Resume a paused expansion process",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def resume_expansion(
    background_tasks: BackgroundTasks,
    expansion_id: str = Path(..., description="Expansion ID"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> ExpansionStatus:
    """Resume a paused expansion process."""
    try:
        if expansion_id not in EXPANSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expansion with ID {expansion_id} not found",
            )
        
        exp = EXPANSIONS[expansion_id]
        
        if exp["status"] != "paused":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expansion is not paused (current status: {exp['status']})",
            )
        
        # Update status
        exp["status"] = "running"
        exp["updated_at"] = datetime.utcnow()
        
        # Resume expansion in background
        background_tasks.add_task(
            run_expansion,
            expansion_id=expansion_id,
            config=ExpansionConfig(**exp["config"]),
            graph_manager=graph_manager,
        )
        
        return ExpansionStatus(
            id=expansion_id,
            config=ExpansionConfig(**exp["config"]),
            current_iteration=exp["current_iteration"],
            total_concepts=exp["total_concepts"],
            total_relationships=exp["total_relationships"],
            status=exp["status"],
            created_at=exp["created_at"],
            updated_at=exp["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error resuming expansion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resuming expansion: {str(e)}",
        )


@router.delete(
    "/{expansion_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete expansion",
    description="Delete an expansion process and its events",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def delete_expansion(
    expansion_id: str = Path(..., description="Expansion ID"),
    api_key: ApiKey = Depends(get_api_key),
) -> None:
    """Delete an expansion process and its events."""
    try:
        if expansion_id not in EXPANSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expansion with ID {expansion_id} not found",
            )
        
        # Delete expansion and events
        del EXPANSIONS[expansion_id]
        if expansion_id in EXPANSION_EVENTS:
            del EXPANSION_EVENTS[expansion_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting expansion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting expansion: {str(e)}",
        )


@router.get(
    "/{expansion_id}/events",
    response_model=List[ExpansionEvent],
    summary="Get expansion events",
    description="Get events for an expansion process",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def get_expansion_events(
    expansion_id: str = Path(..., description="Expansion ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    offset: int = Query(0, ge=0, description="Number of events to skip"),
    api_key: ApiKey = Depends(get_api_key),
) -> List[ExpansionEvent]:
    """Get events for an expansion process."""
    try:
        if expansion_id not in EXPANSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expansion with ID {expansion_id} not found",
            )
        
        if expansion_id not in EXPANSION_EVENTS:
            return []
        
        # Apply pagination
        events = EXPANSION_EVENTS[expansion_id][offset:offset + limit]
        
        return [ExpansionEvent(**event) for event in events]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting expansion events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting expansion events: {str(e)}",
        )


@router.get(
    "/{expansion_id}/stream",
    summary="Stream expansion events",
    description="Stream events for an expansion process in real-time",
    dependencies=[Depends(has_permission(Permission.MANAGE_EXPANSION))],
)
async def stream_expansion_events(
    expansion_id: str = Path(..., description="Expansion ID"),
    api_key: ApiKey = Depends(get_api_key),
):
    """Stream events for an expansion process in real-time."""
    try:
        if expansion_id not in EXPANSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expansion with ID {expansion_id} not found",
            )
        
        # Initialize event index
        if expansion_id not in EXPANSION_EVENTS:
            EXPANSION_EVENTS[expansion_id] = []
        
        event_index = 0
        
        async def event_generator():
            nonlocal event_index
            
            while True:
                # Check if expansion is still active
                if EXPANSIONS[expansion_id]["status"] in ["completed", "failed"]:
                    # Send any remaining events
                    while event_index < len(EXPANSION_EVENTS[expansion_id]):
                        event = EXPANSION_EVENTS[expansion_id][event_index]
                        event_index += 1
                        yield f"data: {json.dumps(event)}\n\n"
                    
                    # Send completion event
                    yield f"data: {json.dumps({'event_type': 'stream_complete'})}\n\n"
                    break
                
                # Send any new events
                while event_index < len(EXPANSION_EVENTS[expansion_id]):
                    event = EXPANSION_EVENTS[expansion_id][event_index]
                    event_index += 1
                    yield f"data: {json.dumps(event)}\n\n"
                
                # Wait for more events
                await asyncio.sleep(1)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error streaming expansion events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error streaming expansion events: {str(e)}",
        )
