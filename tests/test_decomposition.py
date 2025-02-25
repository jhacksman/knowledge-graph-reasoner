"""Tests for query decomposition."""
import pytest
from unittest.mock import AsyncMock, patch
import numpy as np

from src.reasoning.decomposition import QueryDecomposer
from src.reasoning.llm import VeniceLLM, VeniceLLMConfig


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    config = VeniceLLMConfig(api_key="test", model="test")
    llm = VeniceLLM(config)
    return llm


@pytest.mark.asyncio
async def test_decompose_query(mock_llm):
    """Test query decomposition."""
    # Setup mock response
    mock_response = {
        "choices": [{
            "message": {
                "content": "What is the relationship between A and B?\nHow does C affect B?\nWhat properties does A have?"
            }
        }]
    }
    
    # Setup mock LLM
    mock_llm.generate = AsyncMock(return_value=mock_response)
    
    # Test decomposition
    decomposer = QueryDecomposer(mock_llm)
    subqueries = await decomposer.decompose_query(
        "Explain the relationship between A, B, and C"
    )
    
    # Verify results
    assert len(subqueries) == 3
    assert subqueries[0]["text"] == "What is the relationship between A and B?"
    assert subqueries[1]["text"] == "How does C affect B?"
    assert subqueries[2]["text"] == "What properties does A have?"
    assert all(q["type"] == "subquery" for q in subqueries)


@pytest.mark.asyncio
async def test_decompose_query_with_context(mock_llm):
    """Test query decomposition with context."""
    # Setup mock response
    mock_response = {
        "choices": [{
            "message": {
                "content": "What is X in context Y?\nHow does X relate to Z?"
            }
        }]
    }
    
    # Setup mock LLM
    mock_llm.generate = AsyncMock(return_value=mock_response)
    
    # Test decomposition with context
    decomposer = QueryDecomposer(mock_llm)
    subqueries = await decomposer.decompose_query(
        "Explain X in relation to Y and Z",
        context={"domain": "test", "focus": "relationships"}
    )
    
    # Verify results
    assert len(subqueries) == 2
    assert all(q["type"] == "subquery" for q in subqueries)
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_get_query_embedding(mock_llm):
    """Test query embedding."""
    # Setup mock embedding
    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_llm.embed_text = AsyncMock(return_value=mock_embedding)
    
    # Test embedding
    decomposer = QueryDecomposer(mock_llm)
    embedding = await decomposer.get_query_embedding("test query")
    
    # Verify results
    np.testing.assert_array_equal(embedding, mock_embedding)
    mock_llm.embed_text.assert_called_once_with("test query")
