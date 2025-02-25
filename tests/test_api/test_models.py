"""Tests for the API models."""
import pytest
from pydantic import ValidationError
from datetime import datetime
from src.api.models import (
    Concept, ConceptCreate, ConceptUpdate, ConceptList,
    Relationship, RelationshipCreate, RelationshipUpdate, RelationshipList,
    Query, QueryStatus, QueryResult,
    ExpansionConfig, ExpansionStatus, ExpansionEvent,
    MetricsRequest, MetricsResponse, MetricsTimeSeriesResponse, MetricsTimeSeriesPoint,
    SearchRequest, SearchResponse, SearchResult,
    PaginationParams, FilterParams, ErrorResponse
)


def test_concept_create():
    """Test ConceptCreate model."""
    # Test valid data
    concept = ConceptCreate(
        name="Test Concept",
        description="A test concept",
        domain="test",
        attributes={"key": "value"},
    )
    assert concept.name == "Test Concept"
    assert concept.description == "A test concept"
    assert concept.domain == "test"
    assert concept.attributes == {"key": "value"}

    # Test required fields
    with pytest.raises(ValidationError):
        ConceptCreate(
            description="A test concept",
            domain="test",
        )


def test_concept_update():
    """Test ConceptUpdate model."""
    # Test valid data
    concept = ConceptUpdate(
        name="Updated Concept",
        description="An updated concept",
        domain="updated",
        attributes={"key": "updated"},
    )
    assert concept.name == "Updated Concept"
    assert concept.description == "An updated concept"
    assert concept.domain == "updated"
    assert concept.attributes == {"key": "updated"}

    # Test partial update
    concept = ConceptUpdate(
        name="Updated Concept",
    )
    assert concept.name == "Updated Concept"
    assert concept.description is None
    assert concept.domain is None
    assert concept.attributes is None


def test_concept():
    """Test Concept model."""
    # Test valid data
    concept = Concept(
        id="123",
        name="Test Concept",
        description="A test concept",
        domain="test",
        attributes={"key": "value"},
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    assert concept.id == "123"
    assert concept.name == "Test Concept"
    assert concept.description == "A test concept"
    assert concept.domain == "test"
    assert concept.attributes == {"key": "value"}
    assert concept.embedding == [0.1, 0.2, 0.3]
    assert isinstance(concept.created_at, datetime)
    assert isinstance(concept.updated_at, datetime)


def test_concept_list():
    """Test ConceptList model."""
    # Test valid data
    concept = Concept(
        id="123",
        name="Test Concept",
        description="A test concept",
        domain="test",
        attributes={"key": "value"},
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    concept_list = ConceptList(
        items=[concept],
        total=1,
        page=1,
        limit=10,
        pages=1,
    )
    assert len(concept_list.items) == 1
    assert concept_list.items[0].id == "123"
    assert concept_list.total == 1
    assert concept_list.page == 1
    assert concept_list.limit == 10
    assert concept_list.pages == 1


def test_relationship_create():
    """Test RelationshipCreate model."""
    # Test valid data
    relationship = RelationshipCreate(
        source_id="123",
        target_id="456",
        type="related_to",
        weight=0.8,
        attributes={"key": "value"},
    )
    assert relationship.source_id == "123"
    assert relationship.target_id == "456"
    assert relationship.type == "related_to"
    assert relationship.weight == 0.8
    assert relationship.attributes == {"key": "value"}

    # Test required fields
    with pytest.raises(ValidationError):
        RelationshipCreate(
            source_id="123",
            type="related_to",
        )


def test_relationship_update():
    """Test RelationshipUpdate model."""
    # Test valid data
    relationship = RelationshipUpdate(
        type="updated_relation",
        weight=0.9,
        attributes={"key": "updated"},
    )
    assert relationship.type == "updated_relation"
    assert relationship.weight == 0.9
    assert relationship.attributes == {"key": "updated"}

    # Test partial update
    relationship = RelationshipUpdate(
        type="updated_relation",
    )
    assert relationship.type == "updated_relation"
    assert relationship.weight is None
    assert relationship.attributes is None


def test_relationship():
    """Test Relationship model."""
    # Test valid data
    relationship = Relationship(
        id="789",
        source_id="123",
        target_id="456",
        type="related_to",
        weight=0.8,
        attributes={"key": "value"},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    assert relationship.id == "789"
    assert relationship.source_id == "123"
    assert relationship.target_id == "456"
    assert relationship.type == "related_to"
    assert relationship.weight == 0.8
    assert relationship.attributes == {"key": "value"}
    assert isinstance(relationship.created_at, datetime)
    assert isinstance(relationship.updated_at, datetime)


def test_relationship_list():
    """Test RelationshipList model."""
    # Test valid data
    relationship = Relationship(
        id="789",
        source_id="123",
        target_id="456",
        type="related_to",
        weight=0.8,
        attributes={"key": "value"},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    relationship_list = RelationshipList(
        items=[relationship],
        total=1,
        page=1,
        limit=10,
        pages=1,
    )
    assert len(relationship_list.items) == 1
    assert relationship_list.items[0].id == "789"
    assert relationship_list.total == 1
    assert relationship_list.page == 1
    assert relationship_list.limit == 10
    assert relationship_list.pages == 1


def test_query():
    """Test Query model."""
    # Test valid data
    query = Query(
        id="123",
        text="Test query",
        max_results=10,
        include_reasoning=True,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    assert query.id == "123"
    assert query.text == "Test query"
    assert query.max_results == 10
    assert query.include_reasoning is True
    assert query.status == "pending"
    assert isinstance(query.created_at, datetime)
    assert isinstance(query.updated_at, datetime)


def test_query_status():
    """Test QueryStatus model."""
    # Test valid data
    query_status = QueryStatus(
        id="123",
        text="Test query",
        max_results=10,
        include_reasoning=True,
        status="completed",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    assert query_status.id == "123"
    assert query_status.text == "Test query"
    assert query_status.max_results == 10
    assert query_status.include_reasoning is True
    assert query_status.status == "completed"
    assert isinstance(query_status.created_at, datetime)
    assert isinstance(query_status.updated_at, datetime)


def test_query_result():
    """Test QueryResult model."""
    # Test valid data
    concept = Concept(
        id="123",
        name="Test Concept",
        description="A test concept",
        domain="test",
        attributes={"key": "value"},
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    relationship = Relationship(
        id="789",
        source_id="123",
        target_id="456",
        type="related_to",
        weight=0.8,
        attributes={"key": "value"},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    query_result = QueryResult(
        id="123",
        text="Test query",
        concepts=[concept],
        relationships=[relationship],
        reasoning="Test reasoning",
        created_at=datetime.utcnow(),
    )
    assert query_result.id == "123"
    assert query_result.text == "Test query"
    assert len(query_result.concepts) == 1
    assert query_result.concepts[0].id == "123"
    assert len(query_result.relationships) == 1
    assert query_result.relationships[0].id == "789"
    assert query_result.reasoning == "Test reasoning"
    assert isinstance(query_result.created_at, datetime)


def test_expansion_config():
    """Test ExpansionConfig model."""
    # Test valid data
    expansion_config = ExpansionConfig(
        seed_concepts=["123", "456"],
        max_iterations=10,
        expansion_breadth=5,
        domains=["test", "example"],
        checkpoint_interval=2,
    )
    assert expansion_config.seed_concepts == ["123", "456"]
    assert expansion_config.max_iterations == 10
    assert expansion_config.expansion_breadth == 5
    assert expansion_config.domains == ["test", "example"]
    assert expansion_config.checkpoint_interval == 2

    # Test required fields
    with pytest.raises(ValidationError):
        ExpansionConfig(
            max_iterations=10,
            expansion_breadth=5,
        )


def test_expansion_status():
    """Test ExpansionStatus model."""
    # Test valid data
    expansion_config = ExpansionConfig(
        seed_concepts=["123", "456"],
        max_iterations=10,
        expansion_breadth=5,
        domains=["test", "example"],
        checkpoint_interval=2,
    )
    expansion_status = ExpansionStatus(
        id="123",
        config=expansion_config,
        current_iteration=5,
        total_concepts=100,
        total_relationships=200,
        status="running",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    assert expansion_status.id == "123"
    assert expansion_status.config.seed_concepts == ["123", "456"]
    assert expansion_status.current_iteration == 5
    assert expansion_status.total_concepts == 100
    assert expansion_status.total_relationships == 200
    assert expansion_status.status == "running"
    assert isinstance(expansion_status.created_at, datetime)
    assert isinstance(expansion_status.updated_at, datetime)


def test_expansion_event():
    """Test ExpansionEvent model."""
    # Test valid data
    expansion_event = ExpansionEvent(
        expansion_id="123",
        event_type="iteration_complete",
        data={"iteration": 5, "new_concepts": 10},
        timestamp=datetime.utcnow(),
    )
    assert expansion_event.expansion_id == "123"
    assert expansion_event.event_type == "iteration_complete"
    assert expansion_event.data == {"iteration": 5, "new_concepts": 10}
    assert isinstance(expansion_event.timestamp, datetime)


def test_metrics_request():
    """Test MetricsRequest model."""
    # Test valid data
    metrics_request = MetricsRequest(
        metrics=["node_count", "edge_count", "density"],
        from_timestamp=datetime.utcnow(),
        to_timestamp=datetime.utcnow(),
    )
    assert metrics_request.metrics == ["node_count", "edge_count", "density"]
    assert isinstance(metrics_request.from_timestamp, datetime)
    assert isinstance(metrics_request.to_timestamp, datetime)

    # Test required fields
    with pytest.raises(ValidationError):
        MetricsRequest(
            from_timestamp=datetime.utcnow(),
            to_timestamp=datetime.utcnow(),
        )


def test_metrics_response():
    """Test MetricsResponse model."""
    # Test valid data
    metrics_response = MetricsResponse(
        metrics={"node_count": 100, "edge_count": 200, "density": 0.1},
        timestamp=datetime.utcnow(),
    )
    assert metrics_response.metrics == {"node_count": 100, "edge_count": 200, "density": 0.1}
    assert isinstance(metrics_response.timestamp, datetime)


def test_metrics_time_series_point():
    """Test MetricsTimeSeriesPoint model."""
    # Test valid data
    metrics_time_series_point = MetricsTimeSeriesPoint(
        timestamp=datetime.utcnow(),
        value=100,
    )
    assert isinstance(metrics_time_series_point.timestamp, datetime)
    assert metrics_time_series_point.value == 100


def test_metrics_time_series_response():
    """Test MetricsTimeSeriesResponse model."""
    # Test valid data
    point1 = MetricsTimeSeriesPoint(
        timestamp=datetime.utcnow(),
        value=100,
    )
    point2 = MetricsTimeSeriesPoint(
        timestamp=datetime.utcnow(),
        value=200,
    )
    metrics_time_series_response = MetricsTimeSeriesResponse(
        metric="node_count",
        data=[point1, point2],
    )
    assert metrics_time_series_response.metric == "node_count"
    assert len(metrics_time_series_response.data) == 2
    assert metrics_time_series_response.data[0].value == 100
    assert metrics_time_series_response.data[1].value == 200


def test_search_request():
    """Test SearchRequest model."""
    # Test valid data
    search_request = SearchRequest(
        query="test query",
        limit=10,
        threshold=0.7,
        domains=["test", "example"],
    )
    assert search_request.query == "test query"
    assert search_request.limit == 10
    assert search_request.threshold == 0.7
    assert search_request.domains == ["test", "example"]

    # Test required fields
    with pytest.raises(ValidationError):
        SearchRequest(
            limit=10,
            threshold=0.7,
        )


def test_search_result():
    """Test SearchResult model."""
    # Test valid data
    concept = Concept(
        id="123",
        name="Test Concept",
        description="A test concept",
        domain="test",
        attributes={"key": "value"},
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    search_result = SearchResult(
        concept=concept,
        similarity=0.85,
    )
    assert search_result.concept.id == "123"
    assert search_result.similarity == 0.85


def test_search_response():
    """Test SearchResponse model."""
    # Test valid data
    concept = Concept(
        id="123",
        name="Test Concept",
        description="A test concept",
        domain="test",
        attributes={"key": "value"},
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    search_result = SearchResult(
        concept=concept,
        similarity=0.85,
    )
    search_response = SearchResponse(
        results=[search_result],
        query="test query",
        total=1,
    )
    assert len(search_response.results) == 1
    assert search_response.results[0].concept.id == "123"
    assert search_response.query == "test query"
    assert search_response.total == 1


def test_pagination_params():
    """Test PaginationParams model."""
    # Test valid data
    pagination_params = PaginationParams(
        page=2,
        limit=20,
        sort_by="name",
        sort_order="desc",
    )
    assert pagination_params.page == 2
    assert pagination_params.limit == 20
    assert pagination_params.sort_by == "name"
    assert pagination_params.sort_order == "desc"

    # Test default values
    pagination_params = PaginationParams()
    assert pagination_params.page == 1
    assert pagination_params.limit == 10
    assert pagination_params.sort_by == "created_at"
    assert pagination_params.sort_order == "desc"


def test_filter_params():
    """Test FilterParams model."""
    # Test valid data
    filter_params = FilterParams(
        domain="test",
        name_contains="example",
        created_after=datetime.utcnow(),
        created_before=datetime.utcnow(),
    )
    assert filter_params.domain == "test"
    assert filter_params.name_contains == "example"
    assert isinstance(filter_params.created_after, datetime)
    assert isinstance(filter_params.created_before, datetime)


def test_error_response():
    """Test ErrorResponse model."""
    # Test valid data
    error_response = ErrorResponse(
        detail="An error occurred",
        error_code="internal_server_error",
        path="/api/concepts",
        timestamp=datetime.utcnow(),
    )
    assert error_response.detail == "An error occurred"
    assert error_response.error_code == "internal_server_error"
    assert error_response.path == "/api/concepts"
    assert isinstance(error_response.timestamp, datetime)
