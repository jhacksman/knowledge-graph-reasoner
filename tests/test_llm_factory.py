"""Tests for LLM factory."""
import pytest
import pytest_asyncio
import asyncio
import os
from unittest.mock import patch, AsyncMock

from src.reasoning.llm_factory import create_llm
from src.reasoning.llm import VeniceLLM
from src.reasoning.rate_limited_llm import RateLimitedVeniceLLM, RateLimitedVeniceLLMConfig
from src.utils.rate_limiter import RateLimiter


@pytest.mark.asyncio
class TestLLMFactory:
    """Tests for LLM factory."""
    
    async def test_create_llm_with_rate_limiting(self, monkeypatch):
        """Test creating LLM with rate limiting."""
        # Mock RateLimiter to avoid starting the worker task
        original_init = RateLimiter.__init__
        
        def mock_init(self, config):
            self.config = config
            self._queue = asyncio.Queue()
            self._endpoints = {}
            self._storage = None
            # Skip starting the worker task
        
        monkeypatch.setattr(RateLimiter, "__init__", mock_init)
        monkeypatch.setattr(RateLimiter, "close", AsyncMock())
        
        # Set environment variables
        with patch.dict(os.environ, {
            "VENICE_API_KEY": "test_key",
            "VENICE_MODEL": "test_model",
            "VENICE_RATE_LIMIT_CALLS_PER_MINUTE": "20",
            "VENICE_RATE_LIMIT_CALLS_PER_DAY": "5000",
            "VENICE_RATE_LIMIT_TOKENS_PER_MINUTE": "100000"
        }):
            # Create LLM
            llm = create_llm(use_rate_limiting=True)
            
            # Check type
            assert isinstance(llm, RateLimitedVeniceLLM)
            
            # Check config
            assert llm.config.api_key == "test_key"
            assert llm.config.model == "test_model"
            
            # Check rate limiting config
            assert isinstance(llm.config, RateLimitedVeniceLLMConfig)
            rate_limited_config = llm.config
            assert rate_limited_config.calls_per_minute == 20
            assert rate_limited_config.calls_per_day == 5000
            assert rate_limited_config.tokens_per_minute == 100000
            
            # Clean up
            await llm.close()
    
    def test_create_llm_without_rate_limiting(self):
        """Test creating LLM without rate limiting."""
        # Set environment variables
        with patch.dict(os.environ, {
            "VENICE_API_KEY": "test_key",
            "VENICE_MODEL": "test_model"
        }):
            # Create LLM
            llm = create_llm(use_rate_limiting=False)
            
            # Check type
            assert isinstance(llm, VeniceLLM)
            
            # Check config
            assert llm.config.api_key == "test_key"
            assert llm.config.model == "test_model"
    
    def test_create_llm_with_custom_api_key(self):
        """Test creating LLM with custom API key."""
        # Create LLM with custom API key
        llm = create_llm(api_key="custom_key", use_rate_limiting=False)
        
        # Check config
        assert llm.config.api_key == "custom_key"
