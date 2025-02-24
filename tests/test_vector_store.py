"""Tests for vector store implementation."""
import pytest
import numpy as np
from typing import AsyncIterator

from src.models.node import Node
from src.models.edge import Edge
from src.vector_store.milvus_store import MilvusGraphStore


@pytest.fixture
async def milvus_store():
    """Create a test Milvus store."""
    store = MilvusGraphStore(
        uri="./test_milvus.db",
        dim=4,  # Small dimension for testing
        default_collection="test_graph"
    )
    yield store
    # Cleanup
    await store.clear_db("test_graph_nodes")
    await store.clear_db("test_graph_edges")


@pytest.mark.asyncio
async def test_add_and_get_node(milvus_store):
    """Test adding and retrieving a node."""
    node = Node(
        id="test_node_1",
        embedding=np.array([0.1, 0.2, 0.3, 0.4]),
        content="Test content",
        metadata={"test": "metadata"}
    )
    
    # Add node
    node_id = await milvus_store.add_node(node)
    assert node_id == "test_node_1"
    
    # Retrieve node
    retrieved = await milvus_store.get_node(node_id)
    assert retrieved is not None
    assert retrieved.id == node.id
    assert retrieved.content == node.content
    assert retrieved.metadata == node.metadata
    np.testing.assert_array_almost_equal(retrieved.embedding, node.embedding)


@pytest.mark.asyncio
async def test_add_and_get_edge(milvus_store):
    """Test adding and retrieving an edge."""
    edge = Edge(
        source="test_node_1",
        target="test_node_2",
        type="test_relation",
        metadata={"test": "metadata"}
    )
    
    # Add edge
    await milvus_store.add_edge(edge)
    
    # Retrieve edge
    async def collect_edges(iterator: AsyncIterator[Edge]) -> list[Edge]:
        return [edge async for edge in iterator]
    
    edges = await collect_edges(milvus_store.get_edges(
        source_id="test_node_1",
        target_id="test_node_2",
        edge_type="test_relation"
    ))
    
    assert len(edges) == 1
    retrieved = edges[0]
    assert retrieved.source == edge.source
    assert retrieved.target == edge.target
    assert retrieved.type == edge.type
    assert retrieved.metadata == edge.metadata


@pytest.mark.asyncio
async def test_search_similar(milvus_store):
    """Test searching for similar nodes."""
    # Add test nodes
    nodes = [
        Node(
            id=f"test_node_{i}",
            embedding=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
            content=f"Test content {i}",
            metadata={"index": i}
        )
        for i in range(5)
    ]
    
    for node in nodes:
        await milvus_store.add_node(node)
    
    # Search with first node's embedding
    query_embedding = nodes[0].embedding
    results = await milvus_store.search_similar(
        embedding=query_embedding,
        k=3,
        threshold=0.5
    )
    
    assert len(results) > 0
    # First result should be the query node itself
    assert results[0].id == nodes[0].id
