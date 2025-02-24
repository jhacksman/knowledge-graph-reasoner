"""Tests for Venice.ai LLM integration."""
import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import patch, AsyncMock, Mock
import aiohttp

from src.reasoning.llm import VeniceLLM, VeniceLLMConfig


@pytest.fixture
def mock_config():
    """Create mock LLM config."""
    return VeniceLLMConfig(
        api_key="test_key",
        model="test_model",
        base_url="https://api.venice.ai/api/v1"
    )


@pytest_asyncio.fixture
async def mock_session():
    """Create mock aiohttp session."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json = AsyncMock()
    mock_response.raise_for_status = Mock()
    
    # Mock the post method to return a context manager
    async def mock_post(*args, **kwargs):
        return mock_response
    
    mock_session.post = AsyncMock()
    mock_session.post.return_value = mock_response
    
    # Mock ClientSession creation
    with patch("aiohttp.ClientSession", return_value=mock_session):
        yield mock_session, mock_response


@pytest.mark.asyncio
async def test_embed_text(mock_config, mock_session):
    """Test text embedding."""
    mock_session, mock_response = mock_session
    
    # Setup mock response
    mock_embedding = [0.1, 0.2, 0.3]
    response_data = {
        "data": [{
            "embedding": mock_embedding
        }]
    }
    mock_response.json.return_value = response_data
    
    # Test embedding
    llm = VeniceLLM(mock_config)
    embedding = await llm.embed_text("test text")
    await llm.close()
    
    # Verify results
    np.testing.assert_array_equal(embedding, np.array(mock_embedding))
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_generate(mock_config, mock_session):
    """Test text generation."""
    mock_session, mock_response = mock_session
    
    # Setup mock response
    response_data = {
        "choices": [{
            "message": {
                "content": "test response"
            }
        }]
    }
    mock_response.json.return_value = response_data
    
    # Test generation
    llm = VeniceLLM(mock_config)
    response = await llm.generate([
        {"role": "user", "content": "test prompt"}
    ])
    await llm.close()
    
    # Verify results
    assert response == response_data
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling(mock_config, mock_session):
    """Test API error handling."""
    mock_session, mock_response = mock_session
    
    # Setup mock response with error
    request_info = Mock()
    request_info.real_url = "https://api.venice.ai/api/v1/embeddings"
    mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=request_info,
        history=(),
        status=405,
        message="Method Not Allowed"
    )
    
    # Test error handling
    llm = VeniceLLM(mock_config)
    with pytest.raises(aiohttp.ClientResponseError, match="405, message='Method Not Allowed'"):
        await llm.embed_text("test text")
    await llm.close()
    
    # Verify API was called
    mock_session.post.assert_called_once()
