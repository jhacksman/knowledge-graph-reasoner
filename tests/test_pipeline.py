"""Tests for the reasoning pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from src.pipeline.executor import ReasoningPipeline, KnowledgeExpansionResult
from src.models.node import Node
from src.models.edge import Edge


@pytest.fixture
def mock_store():
    """Create a mock vector store."""
    store = AsyncMock()
    store.add_node = AsyncMock(return_value="test_node_1")
    store.add_edge = AsyncMock()
    store.search_similar = AsyncMock(return_value=[
        Node(
            id="existing_node_1",
            embedding=np.array([0.1, 0.2, 0.3, 0.4]),
            content="Existing content",
            metadata={"test": "metadata"}
        )
    ])
    return store


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = AsyncMock()
    llm.decompose_query = AsyncMock(return_value=[
        "What is deep learning?",
        "How does it work?"
    ])
    llm.reason_over_context = AsyncMock(return_value="""
    Deep learning is a subset of machine learning.
    It uses neural networks with multiple layers.
    These networks learn hierarchical representations of data.
    """)
    llm.embed_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3, 0.4]))
    return llm


@pytest.fixture
def pipeline(mock_store, mock_llm):
    """Create a test pipeline with mock components."""
    return ReasoningPipeline(mock_store, mock_llm)


@pytest.mark.asyncio
async def test_extract_knowledge(pipeline, mock_llm):
    """Test knowledge extraction from reasoning."""
    mock_llm.reason_over_context.return_value = """
    {
        "nodes": [
            {"content": "Deep learning", "metadata": {"type": "concept"}},
            {"content": "Neural networks", "metadata": {"type": "technology"}}
        ],
        "edges": [
            {
                "source": "Deep learning",
                "target": "Neural networks",
                "type": "uses",
                "metadata": {"confidence": 0.9}
            }
        ]
    }
    """
    
    nodes, edges = await pipeline._extract_knowledge("Test reasoning")
    
    assert len(nodes) == 2
    assert len(edges) == 1
    assert nodes[0].content == "Deep learning"
    assert edges[0].type == "uses"


@pytest.mark.asyncio
async def test_expand_graph(pipeline, mock_store, mock_llm):
    """Test graph expansion from a query."""
    result = await pipeline.expand_graph(
        query="What is deep learning?",
        max_context_nodes=5,
        similarity_threshold=0.7
    )
    
    assert isinstance(result, KnowledgeExpansionResult)
    assert len(result.query_decomposition) == 2
    assert len(result.reasoning_context) > 0
    assert result.final_reasoning
    assert len(result.new_nodes) > 0
    assert len(result.new_edges) > 0
    
    # Verify component interactions
    mock_llm.decompose_query.assert_called_once()
    mock_llm.reason_over_context.assert_called()
    mock_store.search_similar.assert_called()
    mock_store.add_node.assert_called()
    mock_store.add_edge.assert_called()


@pytest.mark.asyncio
async def test_expand_graph_iteratively(pipeline):
    """Test iterative graph expansion."""
    results = await pipeline.expand_graph_iteratively(
        query="What is deep learning?",
        max_iterations=2
    )
    
    assert isinstance(results, list)
    assert len(results) > 0
    for result in results:
        assert isinstance(result, KnowledgeExpansionResult)
        assert result.query_decomposition
        assert result.final_reasoning
