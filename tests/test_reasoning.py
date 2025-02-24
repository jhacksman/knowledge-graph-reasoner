"""Tests for LLM reasoning components."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any

from src.reasoning.llm import VeniceLLM, VeniceLLMConfig


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    session.post = AsyncMock()
    return session


@pytest.fixture
def mock_response():
    """Create a mock API response."""
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": '["What is deep learning?", "How does it work?"]'
            }
        }]
    })
    return response


@pytest.fixture
async def llm_client(mock_session, mock_response):
    """Create a test LLM client with mocked session."""
    config = VeniceLLMConfig(
        api_key="test_key",
        model="test_model"
    )
    client = VeniceLLM(config)
    mock_session.post.return_value.__aenter__.return_value = mock_response
    client.session = mock_session
    return client


@pytest.mark.asyncio
async def test_decompose_query(llm_client, mock_session, mock_response):
    """Test query decomposition."""
    query = "Explain deep learning"
    sub_queries = await llm_client.decompose_query(query)
    
    assert isinstance(sub_queries, list)
    assert len(sub_queries) == 2
    assert "What is deep learning?" in sub_queries
    
    # Verify API call
    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args
    assert "completions" in call_args[0][0]
    assert query in str(call_args[1]["json"])


@pytest.mark.asyncio
async def test_reason_over_context(llm_client, mock_session):
    """Test reasoning over context."""
    # Setup mock response for reasoning
    reasoning_response = AsyncMock()
    reasoning_response.status = 200
    reasoning_response.json = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Reasoned response"
            }
        }]
    })
    mock_session.post.return_value.__aenter__.return_value = reasoning_response
    
    query = "What is machine learning?"
    context = ["Context 1", "Context 2"]
    response = await llm_client.reason_over_context(query, context)
    
    assert response == "Reasoned response"
    
    # Verify API call
    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args
    assert "completions" in call_args[0][0]
    assert query in str(call_args[1]["json"])
    assert all(ctx in str(call_args[1]["json"]) for ctx in context)


@pytest.mark.asyncio
async def test_rate_limit_retry(llm_client, mock_session):
    """Test retry behavior on rate limit."""
    # Setup responses for rate limit then success
    rate_limit_response = AsyncMock()
    rate_limit_response.status = 429
    
    success_response = AsyncMock()
    success_response.status = 200
    success_response.json = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": '["Success after retry"]'
            }
        }]
    })
    
    mock_session.post.return_value.__aenter__.side_effect = [
        rate_limit_response,
        success_response
    ]
    
    query = "Test query"
    result = await llm_client.decompose_query(query)
    
    assert result == ["Success after retry"]
    assert mock_session.post.call_count == 2
