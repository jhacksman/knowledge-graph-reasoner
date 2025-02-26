"""API routes for graph metrics."""
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
from datetime import datetime, timedelta

from src.api.models import (
    MetricsRequest, MetricsResponse, MetricsTimeSeriesResponse, MetricsTimeSeriesPoint,
    ErrorResponse
)
from src.api.auth import get_api_key, has_permission, Permission, ApiKey
from src.graph.manager import GraphManager
from src.metrics.metrics import MetricsTracker
from src.metrics.graph_metrics import GraphMetricsTracker
from src.metrics.advanced_analytics import AdvancedAnalytics

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/metrics",
    tags=["metrics"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.post(
    "",
    response_model=MetricsResponse,
    summary="Get metrics",
    description="Get current graph metrics",
    dependencies=[Depends(has_permission(Permission.READ_METRICS))],
)
async def get_metrics(
    request: MetricsRequest,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> MetricsResponse:
    """Get current graph metrics."""
    try:
        # Initialize metrics trackers
        metrics_tracker = MetricsTracker()
        graph_metrics_tracker = GraphMetricsTracker()
        advanced_analytics = AdvancedAnalytics(graph_manager)
        
        # Update graph data
        nodes = await graph_manager.get_all_nodes()
        edges = await graph_manager.get_relationships()
        
        await metrics_tracker.update_data(nodes, edges)
        await graph_metrics_tracker.update_data(nodes, edges)
        
        # Compute requested metrics
        metrics = {}
        
        for metric_name in request.metrics:
            if hasattr(metrics_tracker, f"compute_{metric_name}"):
                metrics[metric_name] = await getattr(metrics_tracker, f"compute_{metric_name}")()
            elif hasattr(graph_metrics_tracker, f"compute_{metric_name}"):
                metrics[metric_name] = await getattr(graph_metrics_tracker, f"compute_{metric_name}")()
            elif hasattr(advanced_analytics, f"compute_{metric_name}"):
                metrics[metric_name] = await getattr(advanced_analytics, f"compute_{metric_name}")()
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        return MetricsResponse(
            metrics=metrics,
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.exception(f"Error computing metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing metrics: {str(e)}",
        )


@router.post(
    "/time-series",
    response_model=List[MetricsTimeSeriesResponse],
    summary="Get metrics time series",
    description="Get time series of graph metrics",
    dependencies=[Depends(has_permission(Permission.READ_METRICS))],
)
async def get_metrics_time_series(
    request: MetricsRequest,
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
) -> List[MetricsTimeSeriesResponse]:
    """Get time series of graph metrics."""
    try:
        # Initialize metrics trackers
        metrics_tracker = MetricsTracker()
        graph_metrics_tracker = GraphMetricsTracker()
        
        # Get metrics history
        metrics_history = await metrics_tracker.get_history()
        graph_metrics_history = await graph_metrics_tracker.get_history()
        
        # Filter by time range if provided
        if request.from_timestamp:
            metrics_history = [
                m for m in metrics_history
                if m["timestamp"] >= request.from_timestamp
            ]
            graph_metrics_history = [
                m for m in graph_metrics_history
                if m["timestamp"] >= request.from_timestamp
            ]
        
        if request.to_timestamp:
            metrics_history = [
                m for m in metrics_history
                if m["timestamp"] <= request.to_timestamp
            ]
            graph_metrics_history = [
                m for m in graph_metrics_history
                if m["timestamp"] <= request.to_timestamp
            ]
        
        # Combine histories
        combined_history = metrics_history + graph_metrics_history
        
        # Sort by timestamp
        combined_history.sort(key=lambda m: m["timestamp"])
        
        # Group by metric
        metric_series: dict[str, list[dict[str, Any]]] = {}
        
        for entry in combined_history:
            timestamp = entry["timestamp"]
            
            for metric_name, value in entry["metrics"].items():
                if metric_name in request.metrics:
                    if metric_name not in metric_series:
                        metric_series[metric_name] = []
                    
                    metric_series[metric_name].append({
                        "timestamp": timestamp,
                        "value": value,
                    })
        
        # Create response
        result = []
        
        for metric_name, data in metric_series.items():
            result.append(
                MetricsTimeSeriesResponse(
                    metric=metric_name,
                    data=[
                        MetricsTimeSeriesPoint(
                            timestamp=point["timestamp"],
                            value=point["value"],
                        )
                        for point in data
                    ],
                )
            )
        
        return result
    except Exception as e:
        logger.exception(f"Error computing metrics time series: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing metrics time series: {str(e)}",
        )


@router.get(
    "/available",
    response_model=List[str],
    summary="Get available metrics",
    description="Get a list of available metrics",
    dependencies=[Depends(has_permission(Permission.READ_METRICS))],
)
async def get_available_metrics(
    api_key: ApiKey = Depends(get_api_key),
) -> List[str]:
    """Get a list of available metrics."""
    try:
        # Initialize metrics trackers
        metrics_tracker = MetricsTracker()
        graph_metrics_tracker = GraphMetricsTracker()
        advanced_analytics = AdvancedAnalytics(None)
        
        # Get available metrics
        metrics = []
        
        # Get metrics from MetricsTracker
        for name in dir(metrics_tracker):
            if name.startswith("compute_") and callable(getattr(metrics_tracker, name)):
                metrics.append(name[8:])  # Remove "compute_" prefix
        
        # Get metrics from GraphMetricsTracker
        for name in dir(graph_metrics_tracker):
            if name.startswith("compute_") and callable(getattr(graph_metrics_tracker, name)):
                metrics.append(name[8:])  # Remove "compute_" prefix
        
        # Get metrics from AdvancedAnalytics
        for name in dir(advanced_analytics):
            if name.startswith("compute_") and callable(getattr(advanced_analytics, name)):
                metrics.append(name[8:])  # Remove "compute_" prefix
        
        # Remove duplicates and sort
        metrics = sorted(set(metrics))
        
        return metrics
    except Exception as e:
        logger.exception(f"Error getting available metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting available metrics: {str(e)}",
        )


@router.get(
    "/stream",
    summary="Stream metrics updates",
    description="Stream metrics updates in real-time",
    dependencies=[Depends(has_permission(Permission.READ_METRICS))],
)
async def stream_metrics(
    metrics: List[str] = Query(..., description="Metrics to stream"),
    interval: int = Query(5, ge=1, le=60, description="Update interval in seconds"),
    graph_manager: GraphManager = Depends(),
    api_key: ApiKey = Depends(get_api_key),
):
    """Stream metrics updates in real-time."""
    try:
        # Initialize metrics trackers
        metrics_tracker = MetricsTracker()
        graph_metrics_tracker = GraphMetricsTracker()
        advanced_analytics = AdvancedAnalytics(graph_manager)
        
        async def metrics_generator():
            while True:
                # Update graph data
                nodes = await graph_manager.get_all_nodes()
                edges = await graph_manager.get_relationships()
                
                await metrics_tracker.update_data(nodes, edges)
                await graph_metrics_tracker.update_data(nodes, edges)
                
                # Compute requested metrics
                result = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {},
                }
                
                for metric_name in metrics:
                    if hasattr(metrics_tracker, f"compute_{metric_name}"):
                        result["metrics"][metric_name] = await getattr(metrics_tracker, f"compute_{metric_name}")()
                    elif hasattr(graph_metrics_tracker, f"compute_{metric_name}"):
                        result["metrics"][metric_name] = await getattr(graph_metrics_tracker, f"compute_{metric_name}")()
                    elif hasattr(advanced_analytics, f"compute_{metric_name}"):
                        result["metrics"][metric_name] = await getattr(advanced_analytics, f"compute_{metric_name}")()
                
                # Send metrics
                yield f"data: {json.dumps(result)}\n\n"
                
                # Wait for next update
                await asyncio.sleep(interval)
        
        return StreamingResponse(
            metrics_generator(),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(f"Error streaming metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error streaming metrics: {str(e)}",
        )
