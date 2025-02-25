"""Pydantic models for request/response validation in the API."""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from uuid import UUID


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")
    path: Optional[str] = Field(None, description="Path where the error occurred")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the error")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    limit: int = Field(20, ge=1, le=100, description="Number of items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", description="Sort order (asc or desc)")


class FilterParams(BaseModel):
    """Filter parameters."""
    
    field: str = Field(..., description="Field to filter by")
    operator: str = Field(..., description="Operator (eq, gt, lt, contains, etc.)")
    value: Any = Field(..., description="Value to filter by")


# Concept models
class ConceptBase(BaseModel):
    """Base model for concept operations."""
    
    name: str = Field(..., description="Name of the concept")
    description: Optional[str] = Field(None, description="Description of the concept")
    domain: Optional[str] = Field(None, description="Domain the concept belongs to")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes")


class ConceptCreate(ConceptBase):
    """Model for creating a new concept."""
    
    pass


class ConceptUpdate(BaseModel):
    """Model for updating an existing concept."""
    
    name: Optional[str] = Field(None, description="Name of the concept")
    description: Optional[str] = Field(None, description="Description of the concept")
    domain: Optional[str] = Field(None, description="Domain the concept belongs to")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes")


class Concept(ConceptBase):
    """Model for a concept."""
    
    id: str = Field(..., description="Unique identifier for the concept")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the concept")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        """Pydantic config."""
        
        schema_extra = {
            "example": {
                "id": "c123e4567-e89b-12d3-a456-426614174000",
                "name": "Neural Network",
                "description": "A computational model inspired by the human brain",
                "domain": "Machine Learning",
                "attributes": {"type": "supervised", "layers": 3},
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z"
            }
        }


class ConceptList(BaseModel):
    """Model for a list of concepts with pagination."""
    
    items: List[Concept] = Field(..., description="List of concepts")
    total: int = Field(..., description="Total number of concepts")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Number of items per page")
    pages: int = Field(..., description="Total number of pages")


# Relationship models
class RelationshipBase(BaseModel):
    """Base model for relationship operations."""
    
    source_id: str = Field(..., description="ID of the source concept")
    target_id: str = Field(..., description="ID of the target concept")
    type: str = Field(..., description="Type of relationship")
    weight: Optional[float] = Field(1.0, description="Weight/strength of the relationship")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes")


class RelationshipCreate(RelationshipBase):
    """Model for creating a new relationship."""
    
    pass


class RelationshipUpdate(BaseModel):
    """Model for updating an existing relationship."""
    
    type: Optional[str] = Field(None, description="Type of relationship")
    weight: Optional[float] = Field(None, description="Weight/strength of the relationship")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes")


class Relationship(RelationshipBase):
    """Model for a relationship."""
    
    id: str = Field(..., description="Unique identifier for the relationship")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        """Pydantic config."""
        
        schema_extra = {
            "example": {
                "id": "r123e4567-e89b-12d3-a456-426614174000",
                "source_id": "c123e4567-e89b-12d3-a456-426614174000",
                "target_id": "c223e4567-e89b-12d3-a456-426614174000",
                "type": "is_a",
                "weight": 0.95,
                "attributes": {"confidence": 0.9, "source": "reasoning"},
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z"
            }
        }


class RelationshipList(BaseModel):
    """Model for a list of relationships with pagination."""
    
    items: List[Relationship] = Field(..., description="List of relationships")
    total: int = Field(..., description="Total number of relationships")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Number of items per page")
    pages: int = Field(..., description="Total number of pages")


# Query models
class QueryBase(BaseModel):
    """Base model for query operations."""
    
    query: str = Field(..., description="Query text")
    max_results: Optional[int] = Field(10, description="Maximum number of results to return")
    include_reasoning: Optional[bool] = Field(False, description="Include reasoning in response")


class QueryCreate(QueryBase):
    """Model for creating a new query."""
    
    pass


class QueryResult(BaseModel):
    """Model for a query result."""
    
    concepts: List[Concept] = Field(..., description="Relevant concepts")
    relationships: List[Relationship] = Field(..., description="Relevant relationships")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the results")


class Query(QueryBase):
    """Model for a query."""
    
    id: str = Field(..., description="Unique identifier for the query")
    status: str = Field(..., description="Status of the query (pending, processing, completed, failed)")
    result: Optional[QueryResult] = Field(None, description="Query result")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class QueryList(BaseModel):
    """Model for a list of queries with pagination."""
    
    items: List[Query] = Field(..., description="List of queries")
    total: int = Field(..., description="Total number of queries")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Number of items per page")
    pages: int = Field(..., description="Total number of pages")


# Expansion models
class ExpansionConfig(BaseModel):
    """Configuration for knowledge graph expansion."""
    
    seed_concepts: List[str] = Field(..., description="List of seed concept IDs")
    max_iterations: int = Field(10, description="Maximum number of iterations")
    expansion_breadth: int = Field(5, description="Number of new concepts per iteration")
    domains: Optional[List[str]] = Field(None, description="Domains to focus on")
    checkpoint_interval: Optional[int] = Field(None, description="Checkpoint interval in iterations")


class ExpansionStatus(BaseModel):
    """Status of a knowledge graph expansion."""
    
    id: str = Field(..., description="Unique identifier for the expansion")
    config: ExpansionConfig = Field(..., description="Expansion configuration")
    current_iteration: int = Field(..., description="Current iteration")
    total_concepts: int = Field(..., description="Total number of concepts")
    total_relationships: int = Field(..., description="Total number of relationships")
    status: str = Field(..., description="Status of the expansion (pending, running, paused, completed, failed)")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ExpansionEvent(BaseModel):
    """Event emitted during knowledge graph expansion."""
    
    expansion_id: str = Field(..., description="ID of the expansion")
    event_type: str = Field(..., description="Type of event (new_concept, new_relationship, iteration_complete, etc.)")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(..., description="Event timestamp")


# Metrics models
class MetricsRequest(BaseModel):
    """Request for graph metrics."""
    
    metrics: List[str] = Field(..., description="List of metrics to compute")
    from_timestamp: Optional[datetime] = Field(None, description="Start timestamp for time series")
    to_timestamp: Optional[datetime] = Field(None, description="End timestamp for time series")
    interval: Optional[str] = Field("day", description="Interval for time series (hour, day, week, month)")


class MetricsResponse(BaseModel):
    """Response with graph metrics."""
    
    metrics: Dict[str, Any] = Field(..., description="Computed metrics")
    timestamp: datetime = Field(..., description="Timestamp of computation")


class MetricsTimeSeriesPoint(BaseModel):
    """Point in a metrics time series."""
    
    timestamp: datetime = Field(..., description="Timestamp")
    value: Any = Field(..., description="Metric value")


class MetricsTimeSeriesResponse(BaseModel):
    """Response with time series of graph metrics."""
    
    metric: str = Field(..., description="Metric name")
    data: List[MetricsTimeSeriesPoint] = Field(..., description="Time series data")


# Search models
class SearchRequest(BaseModel):
    """Request for semantic search."""
    
    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Maximum number of results")
    threshold: Optional[float] = Field(0.7, description="Similarity threshold")
    domains: Optional[List[str]] = Field(None, description="Domains to search in")


class SearchResult(BaseModel):
    """Result of a semantic search."""
    
    concept: Concept = Field(..., description="Matching concept")
    similarity: float = Field(..., description="Similarity score")


class SearchResponse(BaseModel):
    """Response with search results."""
    
    results: List[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    total: int = Field(..., description="Total number of results")
