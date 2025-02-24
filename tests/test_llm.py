"""Tests for Venice.ai LLM integration."""
import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import patch, AsyncMock
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
    with patch("aiohttp.ClientSession") as mock:
        mock_session = AsyncMock()
        mock.return_value = mock_session
        yield mock_session


@pytest.mark.asyncio
async def test_embed_text(mock_config, mock_session):
    """Test text embedding."""
    # Setup mock response
    mock_embedding = [0.1, 0.2, 0.3]
    mock_response = {
        "data": [{
            "embedding": mock_embedding
        }]
    }
    
    # Setup mock response object
    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()
    mock_response_obj.close = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response_obj)
    
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
    # Setup mock response
    mock_response = {
        "choices": [{
            "message": {
                "content": "test response"
            }
        }]
    }
    
    # Setup mock response object
    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()
    mock_response_obj.close = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response_obj)
    
    # Test generation
    llm = VeniceLLM(mock_config)
    response = await llm.generate([
        {"role": "user", "content": "test prompt"}
    ])
    await llm.close()
    
    # Verify results
    assert response == mock_response
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling(mock_config, mock_session):
    """Test API error handling."""
    # Setup mock response with error
    mock_response_obj = AsyncMock()
    mock_response_obj.raise_for_status = AsyncMock(side_effect=Exception("API Error"))
    mock_response_obj.close = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response_obj)
    
    # Test error handling
    llm = VeniceLLM(mock_config)
    with pytest.raises(Exception, match="API Error"):
        await llm.embed_text("test text")
    await llm.close()
    
    # Verify API was called
    mock_session.post.assert_called_once()
