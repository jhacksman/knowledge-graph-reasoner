"""Tests for query execution pipeline."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np
from typing import List, Optional, AsyncIterator, Dict, Any

from src.pipeline.query_executor import QueryExecutor
from src.reasoning.llm import VeniceLLM, VeniceLLMConfig
from src.models.node import Node
from src.models.edge import Edge
from src.vector_store.base import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing."""
    
    async def initialize(self) -> None:
        pass
    
    async def add_node(self, node: Node) -> str:
        return node.id
    
    async def add_edge(self, edge: Edge) -> None:
        pass
    
    async def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.5,
        deduplicate: bool = True
    ) -> List[Node]:
        return [
            Node(id="test1", content="Test content 1"),
            Node(id="test2", content="Test content 2")
        ]
    
    async def get_node(self, node_id: str) -> Optional[Node]:
        return Node(id=node_id, content=f"Test content for {node_id}")
    
    async def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None
    ) -> AsyncIterator[Edge]:
        edges = [
            Edge(source="test1", target="test2", type="related"),
            Edge(source="test2", target="test3", type="similar")
        ]
        for edge in edges:
            yield edge


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    config = VeniceLLMConfig(api_key="test", model="test")
    llm = VeniceLLM(config)
    return llm


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    return MockVectorStore()


@pytest.mark.asyncio
async def test_execute_query(mock_llm, mock_vector_store):
    """Test query execution."""
    # Setup mock responses
    mock_subqueries = [
        {"text": "What is A?", "type": "subquery", "metadata": {}},
        {"text": "How does A relate to B?", "type": "subquery", "metadata": {}}
    ]
    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_final_response = {
        "choices": [{
            "message": {
                "content": "Test explanation"
            }
        }]
    }
    
    # Setup mock LLM
    mock_llm.generate = AsyncMock()
    mock_llm.generate.side_effect = [
        {"choices": [{"message": {"content": "\n".join(q["text"] for q in mock_subqueries)}}]},
        mock_final_response
    ]
    mock_llm.embed_text = AsyncMock(return_value=mock_embedding)
    
    # Test execution
    executor = QueryExecutor(mock_llm, mock_vector_store)
    result = await executor.execute_query(
        "Test query",
        context={"domain": "test"}
    )
    
    # Verify results
    assert result["query"] == "Test query"
    assert len(result["subqueries"]) == 2
    assert result["explanation"] == "Test explanation"
    assert len(result["results"]) == 2
    for subresult in result["results"]:
        assert len(subresult["nodes"]) == 2
        assert len(list(subresult["edges"])) == 2


@pytest.mark.asyncio
async def test_execute_query_with_error(mock_llm, mock_vector_store):
    """Test query execution with error handling."""
    # Setup mock LLM to raise error
    mock_llm.generate = AsyncMock(side_effect=Exception("API Error"))
    
    # Test execution
    executor = QueryExecutor(mock_llm, mock_vector_store)
    with pytest.raises(Exception, match="API Error"):
        await executor.execute_query("Test query")
