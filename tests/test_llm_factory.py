"""Tests for LLM factory."""
import pytest
import os
from unittest.mock import patch

from src.reasoning.llm_factory import create_llm
from src.reasoning.llm import VeniceLLM
from src.reasoning.rate_limited_llm import RateLimitedVeniceLLM


class TestLLMFactory:
    """Tests for LLM factory."""
    
    def test_create_llm_with_rate_limiting(self):
        """Test creating LLM with rate limiting."""
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
            assert llm.config.calls_per_minute == 20
            assert llm.config.calls_per_day == 5000
            assert llm.config.tokens_per_minute == 100000
    
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
