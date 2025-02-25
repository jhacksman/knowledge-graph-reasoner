"""Factory for creating LLM instances."""
import os
from typing import Optional

from ..reasoning.llm import VeniceLLM, VeniceLLMConfig
from ..reasoning.rate_limited_llm import RateLimitedVeniceLLM, RateLimitedVeniceLLMConfig


def create_venice_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[int] = None,
    rate_limited: bool = True,
    calls_per_minute: Optional[int] = None,
    burst_limit: Optional[int] = None,
    retry_interval: Optional[float] = None
):
    """Create a Venice.ai LLM client.
    
    Args:
        api_key: Venice.ai API key (defaults to VENICE_API_KEY env var)
        model: Model name (defaults to VENICE_MODEL env var or "deepseek-r1-671b")
        base_url: Base API URL (defaults to VENICE_BASE_URL env var)
        max_retries: Maximum number of retries (defaults to 3)
        timeout: Request timeout in seconds (defaults to 30)
        rate_limited: Whether to use rate limiting (defaults to True)
        calls_per_minute: Maximum API calls per minute (defaults to VENICE_RATE_LIMIT_CALLS_PER_MINUTE env var or 60)
        burst_limit: Maximum burst of requests allowed (defaults to VENICE_RATE_LIMIT_BURST_LIMIT env var or 10)
        retry_interval: Time to wait between retries in seconds (defaults to VENICE_RATE_LIMIT_RETRY_INTERVAL env var or 1.0)
        
    Returns:
        VeniceLLM or RateLimitedVeniceLLM: LLM client
    """
    # Get values from environment variables if not provided
    api_key = api_key or os.environ.get("VENICE_API_KEY")
    model = model or os.environ.get("VENICE_MODEL", "deepseek-r1-671b")
    base_url = base_url or os.environ.get("VENICE_BASE_URL", "https://api.venice.ai/api/v1")
    max_retries = max_retries if max_retries is not None else int(os.environ.get("VENICE_MAX_RETRIES", "3"))
    timeout = timeout if timeout is not None else int(os.environ.get("VENICE_TIMEOUT", "30"))
    
    if not api_key:
        raise ValueError("Venice.ai API key is required")
    
    if rate_limited:
        # Get rate limit values from environment variables if not provided
        calls_per_minute = calls_per_minute if calls_per_minute is not None else int(
            os.environ.get("VENICE_RATE_LIMIT_CALLS_PER_MINUTE", "60")
        )
        burst_limit = burst_limit if burst_limit is not None else int(
            os.environ.get("VENICE_RATE_LIMIT_BURST_LIMIT", "10")
        )
        retry_interval = retry_interval if retry_interval is not None else float(
            os.environ.get("VENICE_RATE_LIMIT_RETRY_INTERVAL", "1.0")
        )
        
        config = RateLimitedVeniceLLMConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            calls_per_minute=calls_per_minute,
            burst_limit=burst_limit,
            retry_interval=retry_interval
        )
        return RateLimitedVeniceLLM(config)
    else:
        config = VeniceLLMConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout
        )
        return VeniceLLM(config)
