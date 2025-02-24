"""Tests for Milvus vector store."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.vector_store.milvus_store import (
    MilvusStore,
    MilvusError,
    CollectionInitError,
    SearchError
)
from src.models.node import Node
from src.models.edge import Edge


@pytest.fixture
def mock_milvus_client():
    """Create mock Milvus client."""
    with patch("src.vector_store.milvus_store.MilvusClient") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
async def milvus_store(mock_milvus_client):
    """Create Milvus store with mock client."""
    store = MilvusStore(
        uri="test_uri",
        token="test_token",
        db="test_db",
        default_collection="test_collection"
    )
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_initialize(mock_milvus_client):
    """Test store initialization."""
    store = MilvusStore()
    await store.initialize()
    
    # Verify collections were created
    assert mock_milvus_client.create_collection.call_count == 2
    assert mock_milvus_client.has_collection.call_count == 2


@pytest.mark.asyncio
async def test_add_node(milvus_store, mock_milvus_client):
    """Test adding a node."""
    node = Node(
        id="test_id",
        content="test content",
        metadata={"key": "value"}
    )
    node_id = await milvus_store.add_node(node)
    
    # Verify node was added
    assert node_id == "test_id"
    mock_milvus_client.insert.assert_called_once()


@pytest.mark.asyncio
async def test_add_edge(milvus_store, mock_milvus_client):
    """Test adding an edge."""
    edge = Edge(
        source="source_id",
        target="target_id",
        type="test_type",
        metadata={"key": "value"}
    )
    await milvus_store.add_edge(edge)
    
    # Verify edge was added
    mock_milvus_client.insert.assert_called_once()


@pytest.mark.asyncio
async def test_search_similar(milvus_store, mock_milvus_client):
    """Test searching for similar nodes."""
    # Setup mock search results
    mock_result = MagicMock()
    mock_result.distance = 0.5
    mock_result.entity = {
        "id": "test_id",
        "content": "test content",
        "metadata": {"key": "value"}
    }
    mock_milvus_client.search.return_value = [[mock_result]]
    
    # Search for similar nodes
    embedding = np.random.rand(1536)
    nodes = await milvus_store.search_similar(
        embedding,
        k=5,
        threshold=0.5
    )
    
    # Verify search results
    assert len(nodes) == 1
    assert nodes[0].id == "test_id"
    assert nodes[0].content == "test content"
    mock_milvus_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_get_node(milvus_store, mock_milvus_client):
    """Test getting a node by ID."""
    # Setup mock query result
    mock_milvus_client.query.return_value = [{
        "id": "test_id",
        "content": "test content",
        "metadata": {"key": "value"}
    }]
    
    # Get node
    node = await milvus_store.get_node("test_id")
    
    # Verify node
    assert node is not None
    assert node.id == "test_id"
    assert node.content == "test content"
    mock_milvus_client.query.assert_called_once()


@pytest.mark.asyncio
async def test_get_edges(milvus_store, mock_milvus_client):
    """Test getting edges."""
    # Setup mock query results
    mock_milvus_client.query.return_value = [{
        "source": "source_id",
        "target": "target_id",
        "type": "test_type",
        "metadata": {"key": "value"}
    }]
    
    # Get edges
    edges = []
    async for edge in milvus_store.get_edges(
        source_id="source_id",
        target_id="target_id",
        edge_type="test_type"
    ):
        edges.append(edge)
    
    # Verify edges
    assert len(edges) == 1
    assert edges[0].source == "source_id"
    assert edges[0].target == "target_id"
    mock_milvus_client.query.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling(milvus_store, mock_milvus_client):
    """Test error handling."""
    # Setup mock to raise error
    mock_milvus_client.search.side_effect = Exception("Test error")
    
    # Verify search error is raised
    with pytest.raises(SearchError):
        await milvus_store.search_similar(np.random.rand(1536))
