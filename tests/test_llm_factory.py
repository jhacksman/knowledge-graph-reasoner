"""Tests for LLM factory."""
import pytest
import os
from unittest.mock import patch

from src.factory.llm_factory import create_venice_llm
from src.reasoning.llm import VeniceLLM
from src.reasoning.rate_limited_llm import RateLimitedVeniceLLM


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables."""
    with patch.dict(os.environ, {
        "VENICE_API_KEY": "test_api_key",
        "VENICE_MODEL": "test_model",
        "VENICE_BASE_URL": "https://test.api.venice.ai",
        "VENICE_MAX_RETRIES": "5",
        "VENICE_TIMEOUT": "60",
        "VENICE_RATE_LIMIT_CALLS_PER_MINUTE": "120",
        "VENICE_RATE_LIMIT_BURST_LIMIT": "20",
        "VENICE_RATE_LIMIT_RETRY_INTERVAL": "2.0"
    }):
        yield


def test_create_venice_llm_with_rate_limiting(mock_env_vars):
    """Test creating a rate-limited Venice.ai LLM client."""
    llm = create_venice_llm()
    
    assert isinstance(llm, RateLimitedVeniceLLM)
    assert llm.config.api_key == "test_api_key"
    assert llm.config.model == "test_model"
    assert llm.config.base_url == "https://test.api.venice.ai"
    assert llm.config.max_retries == 5
    assert llm.config.timeout == 60
    assert llm.config.calls_per_minute == 120
    assert llm.config.burst_limit == 20
    assert llm.config.retry_interval == 2.0


def test_create_venice_llm_without_rate_limiting(mock_env_vars):
    """Test creating a Venice.ai LLM client without rate limiting."""
    llm = create_venice_llm(rate_limited=False)
    
    assert isinstance(llm, VeniceLLM)
    assert llm.config.api_key == "test_api_key"
    assert llm.config.model == "test_model"
    assert llm.config.base_url == "https://test.api.venice.ai"
    assert llm.config.max_retries == 5
    assert llm.config.timeout == 60


def test_create_venice_llm_with_explicit_params():
    """Test creating a Venice.ai LLM client with explicit parameters."""
    llm = create_venice_llm(
        api_key="explicit_api_key",
        model="explicit_model",
        base_url="https://explicit.api.venice.ai",
        max_retries=10,
        timeout=120,
        calls_per_minute=240,
        burst_limit=30,
        retry_interval=3.0
    )
    
    assert isinstance(llm, RateLimitedVeniceLLM)
    assert llm.config.api_key == "explicit_api_key"
    assert llm.config.model == "explicit_model"
    assert llm.config.base_url == "https://explicit.api.venice.ai"
    assert llm.config.max_retries == 10
    assert llm.config.timeout == 120
    assert llm.config.calls_per_minute == 240
    assert llm.config.burst_limit == 30
    assert llm.config.retry_interval == 3.0


def test_create_venice_llm_missing_api_key():
    """Test creating a Venice.ai LLM client with missing API key."""
    with pytest.raises(ValueError, match="Venice.ai API key is required"):
        create_venice_llm(api_key=None)
