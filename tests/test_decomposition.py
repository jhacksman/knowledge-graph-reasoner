"""Tests for graph query decomposition."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.reasoning.decomposition import GraphQueryDecomposer
from src.reasoning.llm import VeniceLLM


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = AsyncMock(spec=VeniceLLM)
    response = MagicMock()
    response.content = '["What are the key concepts?", "How do they relate?"]'
    llm.generate = AsyncMock(return_value=response)
    return llm


@pytest.mark.asyncio
async def test_decompose_query(mock_llm):
    """Test basic query decomposition."""
    decomposer = GraphQueryDecomposer(mock_llm)
    query = "Explain deep learning"
    sub_queries = await decomposer.decompose_query(query)
    
    assert isinstance(sub_queries, list)
    assert len(sub_queries) == 2
    assert "key concepts" in sub_queries[0].lower()
    assert "relate" in sub_queries[1].lower()
    
    # Verify prompt structure
    call_args = mock_llm.generate.call_args
    assert query in str(call_args[1]["messages"][0]["content"])


@pytest.mark.asyncio
async def test_decompose_query_with_context(mock_llm):
    """Test query decomposition with context."""
    decomposer = GraphQueryDecomposer(mock_llm)
    query = "Explain deep learning"
    context = ["Neural networks are fundamental", "Backpropagation is key"]
    sub_queries = await decomposer.decompose_query(query, context)
    
    assert isinstance(sub_queries, list)
    # Verify context was included in prompt
    call_args = mock_llm.generate.call_args
    prompt = call_args[1]["messages"][0]["content"]
    assert all(ctx in prompt for ctx in context)


@pytest.mark.asyncio
async def test_decompose_query_error_handling(mock_llm):
    """Test error handling in query decomposition."""
    # Setup mock to raise exception
    mock_llm.generate = AsyncMock(side_effect=Exception("API error"))
    
    decomposer = GraphQueryDecomposer(mock_llm)
    query = "Explain deep learning"
    sub_queries = await decomposer.decompose_query(query)
    
    # Should fall back to original query
    assert sub_queries == [query]
