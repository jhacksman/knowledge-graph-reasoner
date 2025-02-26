"""Tests for rate-limited Venice.ai LLM integration."""
import pytest
import pytest_asyncio
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock

from src.reasoning.rate_limited_llm import RateLimitedVeniceLLM, RateLimitedVeniceLLMConfig
from src.utils.rate_limiter import RateLimiter


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return RateLimitedVeniceLLMConfig(
        api_key="test_key",
        model="test_model",
        calls_per_minute=15,
        calls_per_day=10000,
        tokens_per_minute=200000,
        storage_path=".test_rate_limit_storage.db"
    )


@pytest_asyncio.fixture
async def rate_limited_llm(test_config):
    """Create a rate-limited LLM instance for testing."""
    llm = RateLimitedVeniceLLM(test_config)
    yield llm
    await llm.close()


@pytest.mark.asyncio
class TestRateLimitedVeniceLLM:
    """Tests for RateLimitedVeniceLLM."""
    
    async def test_init(self, test_config):
        """Test initialization."""
        llm = RateLimitedVeniceLLM(test_config)
        
        # Check that rate limiter was created
        assert isinstance(llm.rate_limiter, RateLimiter)
        
        # Check that token counter was created
        assert llm.token_counter is not None
        
        await llm.close()
    
    async def test_close(self, rate_limited_llm):
        """Test closing the client."""
        # Mock rate limiter close method
        rate_limited_llm.rate_limiter.close = AsyncMock()
        
        # Close client
        await rate_limited_llm.close()
        
        # Check that rate limiter close was called
        rate_limited_llm.rate_limiter.close.assert_called_once()
    
    async def test_embed_text(self, rate_limited_llm):
        """Test text embedding with rate limiting."""
        # Create a mock response
        mock_response = np.array([0.1, 0.2, 0.3])
        
        # Mock the parent class embed_text method
        with patch('src.reasoning.llm.VeniceLLM.embed_text', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = mock_response
            
            # Mock rate limiter execute method
            rate_limited_llm.rate_limiter.execute = AsyncMock()
            rate_limited_llm.rate_limiter.execute.return_value = mock_response
            
            # Embed text
            result = await rate_limited_llm.embed_text("test text")
            
            # Check result
            assert isinstance(result, np.ndarray)
            assert result.shape == (3,)
            
            # Check that rate limiter execute was called with correct arguments
            rate_limited_llm.rate_limiter.execute.assert_called_once()
            args, kwargs = rate_limited_llm.rate_limiter.execute.call_args
            assert kwargs["endpoint"] == "embeddings"
            assert kwargs["tokens"] > 0
            assert kwargs["urgent"] is True
    
    async def test_generate(self, rate_limited_llm):
        """Test text generation with rate limiting."""
        # Create a mock response
        mock_response = {"choices": [{"message": {"content": "test response"}}]}
        
        # Mock the parent class generate method
        with patch('src.reasoning.llm.VeniceLLM.generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response
            
            # Mock rate limiter execute method
            rate_limited_llm.rate_limiter.execute = AsyncMock()
            rate_limited_llm.rate_limiter.execute.return_value = mock_response
            
            # Generate text
            messages = [{"role": "user", "content": "test prompt"}]
            result = await rate_limited_llm.generate(messages)
            
            # Check result
            assert isinstance(result, dict)
            assert "choices" in result
            
            # Check that rate limiter execute was called with correct arguments
            rate_limited_llm.rate_limiter.execute.assert_called_once()
            args, kwargs = rate_limited_llm.rate_limiter.execute.call_args
            assert kwargs["endpoint"] == "chat/completions"
            assert kwargs["tokens"] > 0
            assert kwargs["urgent"] is True
