"""Tests for rate limiter."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import time

from src.utils.rate_limiter import RateLimiter
from src.reasoning.rate_limited_llm import RateLimitedVeniceLLM, RateLimitedVeniceLLMConfig


@pytest.fixture
def rate_limiter():
    """Create a rate limiter for testing."""
    return RateLimiter(calls_per_minute=60, burst_limit=5, retry_interval=0.1, max_retries=2)


@pytest.mark.asyncio
async def test_rate_limiter_token_consumption(rate_limiter):
    """Test that tokens are consumed correctly."""
    # Should be able to consume burst_limit tokens immediately
    for _ in range(rate_limiter.burst_limit):
        assert await rate_limiter._consume_token() is True
    
    # Next token should not be available
    assert await rate_limiter._consume_token() is False


@pytest.mark.asyncio
async def test_rate_limiter_token_refill(rate_limiter):
    """Test that tokens are refilled correctly."""
    # Consume all tokens
    for _ in range(rate_limiter.burst_limit):
        assert await rate_limiter._consume_token() is True
    
    # Wait for token refill (should get 1 token after 1 second at 60/min rate)
    await asyncio.sleep(1.1)
    assert await rate_limiter._consume_token() is True
    assert await rate_limiter._consume_token() is False


@pytest.mark.asyncio
async def test_rate_limiter_execute_success(rate_limiter):
    """Test successful execution with rate limiting."""
    mock_func = AsyncMock(return_value="success")
    
    # Should be able to execute burst_limit times
    for _ in range(rate_limiter.burst_limit):
        result = await rate_limiter.execute(mock_func, "arg1", kwarg1="value1")
        assert result == "success"
    
    # Verify metrics
    metrics = rate_limiter.get_metrics()
    assert metrics["total_calls"] == rate_limiter.burst_limit
    assert metrics["successful_calls"] == rate_limiter.burst_limit
    assert metrics["rate_limited_calls"] == 0
    assert metrics["failed_calls"] == 0


@pytest.mark.asyncio
async def test_rate_limiter_execute_rate_limited(rate_limiter):
    """Test execution when rate limited."""
    mock_func = AsyncMock(return_value="success")
    
    # Consume all tokens
    for _ in range(rate_limiter.burst_limit):
        await rate_limiter._consume_token()
    
    # Execute should wait for token and succeed
    with patch.object(rate_limiter, 'wait_for_token', side_effect=[False, True]):
        result = await rate_limiter.execute(mock_func, "arg1")
        assert result == "success"
    
    # Verify metrics
    metrics = rate_limiter.get_metrics()
    assert metrics["total_calls"] == 1
    assert metrics["successful_calls"] == 1
    assert metrics["rate_limited_calls"] == 1
    assert metrics["retried_calls"] == 1


@pytest.mark.asyncio
async def test_rate_limiter_execute_exception(rate_limiter):
    """Test execution when function raises exception."""
    mock_func = AsyncMock(side_effect=Exception("Test error"))
    
    # Execute should raise the exception
    with pytest.raises(Exception, match="Test error"):
        await rate_limiter.execute(mock_func)
    
    # Verify metrics
    metrics = rate_limiter.get_metrics()
    assert metrics["total_calls"] == 1
    assert metrics["successful_calls"] == 0
    assert metrics["failed_calls"] == 1


@pytest.mark.asyncio
async def test_rate_limiter_execute_rate_limit_exception(rate_limiter):
    """Test execution when function raises rate limit exception."""
    mock_func = AsyncMock(side_effect=[
        Exception("Rate limit exceeded"),
        "success"
    ])
    
    # Execute should retry and succeed
    result = await rate_limiter.execute(mock_func)
    assert result == "success"
    
    # Verify metrics
    metrics = rate_limiter.get_metrics()
    assert metrics["total_calls"] == 1
    assert metrics["successful_calls"] == 1
    assert metrics["rate_limited_calls"] == 1
    assert metrics["retried_calls"] == 1


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return RateLimitedVeniceLLMConfig(
        api_key="test_key",
        model="test_model",
        calls_per_minute=60,
        burst_limit=5
    )


@pytest.mark.asyncio
async def test_rate_limited_llm_init(mock_config):
    """Test initialization of rate limited LLM."""
    llm = RateLimitedVeniceLLM(mock_config)
    assert llm.config.api_key == "test_key"
    assert llm.config.model == "test_model"
    assert llm.config.calls_per_minute == 60
    assert llm.config.burst_limit == 5
    assert isinstance(llm.rate_limiter, RateLimiter)


@pytest.mark.asyncio
async def test_rate_limited_llm_embed_text(mock_config):
    """Test embed_text with rate limiting."""
    llm = RateLimitedVeniceLLM(mock_config)
    
    # Mock the parent class embed_text method
    with patch.object(llm.__class__.__mro__[1], 'embed_text', AsyncMock()) as mock_embed:
        await llm.embed_text("test text")
        mock_embed.assert_called_once_with("test text")
    
    # Verify metrics
    metrics = llm.get_rate_limit_metrics()
    assert metrics["total_calls"] == 1
    assert metrics["successful_calls"] == 1


@pytest.mark.asyncio
async def test_rate_limited_llm_generate(mock_config):
    """Test generate with rate limiting."""
    llm = RateLimitedVeniceLLM(mock_config)
    
    # Mock the parent class generate method
    with patch.object(llm.__class__.__mro__[1], 'generate', AsyncMock()) as mock_generate:
        messages = [{"role": "user", "content": "Hello"}]
        await llm.generate(messages, temperature=0.5, max_tokens=100)
        mock_generate.assert_called_once_with(
            messages, temperature=0.5, max_tokens=100
        )
    
    # Verify metrics
    metrics = llm.get_rate_limit_metrics()
    assert metrics["total_calls"] == 1
    assert metrics["successful_calls"] == 1
