"""Factory for creating LLM instances."""
import os
from typing import Optional, Union, cast
# mypy: ignore-errors
from dotenv import load_dotenv  # type: ignore

from .llm import VeniceLLM, VeniceLLMConfig
from .rate_limited_llm import RateLimitedVeniceLLM, RateLimitedVeniceLLMConfig

# Load environment variables
load_dotenv()  # type: ignore

def create_llm(api_key: Optional[str] = None, use_rate_limiting: bool = True) -> Union[VeniceLLM, RateLimitedVeniceLLM]:
    """Create an LLM instance.
    
    Args:
        api_key: Venice.ai API key. If None, will use VENICE_API_KEY from environment.
        use_rate_limiting: Whether to use rate limiting.
        
    Returns:
        Union[VeniceLLM, RateLimitedVeniceLLM]: LLM instance.
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("VENICE_API_KEY", "")
    
    # Create base config
    base_config = VeniceLLMConfig(
        api_key=api_key,
        model=os.environ.get("VENICE_MODEL", "deepseek-r1-671b"),
        base_url=os.environ.get("VENICE_BASE_URL", "https://api.venice.ai/api/v1")
    )
    
    # Create LLM with or without rate limiting
    if use_rate_limiting:
        # Create rate-limited config by extending base config
        config = RateLimitedVeniceLLMConfig(
            # Pass base config parameters
            api_key=base_config.api_key,
            model=base_config.model,
            base_url=base_config.base_url,
            max_retries=base_config.max_retries,
            timeout=base_config.timeout,
            # Add rate limiting parameters
            calls_per_minute=int(os.environ.get("VENICE_RATE_LIMIT_CALLS_PER_MINUTE", "15")),
            calls_per_day=int(os.environ.get("VENICE_RATE_LIMIT_CALLS_PER_DAY", "10000")),
            tokens_per_minute=int(os.environ.get("VENICE_RATE_LIMIT_TOKENS_PER_MINUTE", "200000")),
            burst_limit=int(os.environ.get("VENICE_RATE_LIMIT_BURST_LIMIT", "5")),
            retry_interval=float(os.environ.get("VENICE_RATE_LIMIT_RETRY_INTERVAL", "1.0")),
            jitter_factor=float(os.environ.get("VENICE_RATE_LIMIT_JITTER_FACTOR", "0.1")),
            storage_path=os.environ.get("VENICE_RATE_LIMIT_STORAGE_PATH", ".rate_limit_storage.db"),
            queue_size=int(os.environ.get("VENICE_RATE_LIMIT_QUEUE_SIZE", "100")),
            non_urgent_timeout=float(os.environ.get("VENICE_RATE_LIMIT_NON_URGENT_TIMEOUT", "60.0"))
        )
        return RateLimitedVeniceLLM(config)
    else:
        return VeniceLLM(base_config)
