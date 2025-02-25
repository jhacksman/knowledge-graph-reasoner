"""Rate-limited Venice.ai LLM integration."""
from typing import Dict, Any, List, Optional
import numpy as np
import logging

from .llm import VeniceLLM, VeniceLLMConfig
from ..utils.rate_limiter import RateLimiter

log = logging.getLogger(__name__)


class RateLimitedVeniceLLMConfig(VeniceLLMConfig):
    """Configuration for rate-limited Venice.ai LLM client."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-r1-671b",
        base_url: str = "https://api.venice.ai/api/v1",
        max_retries: int = 3,
        timeout: int = 30,
        calls_per_minute: int = 60,
        burst_limit: int = 10,
        retry_interval: float = 1.0
    ):
        """Initialize rate-limited LLM config.
        
        Args:
            api_key: Venice.ai API key
            model: Model name to use
            base_url: Base API URL
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            calls_per_minute: Maximum number of calls allowed per minute
            burst_limit: Maximum number of calls allowed in a burst
            retry_interval: Time to wait between retries in seconds
        """
        super().__init__(api_key, model, base_url, max_retries, timeout)
        self.calls_per_minute = calls_per_minute
        self.burst_limit = burst_limit
        self.retry_interval = retry_interval


class RateLimitedVeniceLLM(VeniceLLM):
    """Rate-limited Venice.ai LLM client."""
    
    def __init__(self, config: RateLimitedVeniceLLMConfig):
        """Initialize rate-limited LLM client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        self.rate_limiter = RateLimiter(
            calls_per_minute=config.calls_per_minute,
            burst_limit=config.burst_limit,
            retry_interval=config.retry_interval,
            max_retries=config.max_retries
        )
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Get text embedding with rate limiting.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Text embedding
        """
        return await self.rate_limiter.execute(super().embed_text, text)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate text completion with rate limiting.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict[str, Any]: Response data
        """
        return await self.rate_limiter.execute(
            super().generate,
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def get_rate_limit_metrics(self) -> Dict[str, int]:
        """Get rate limiter metrics.
        
        Returns:
            Dict[str, int]: Metrics dictionary
        """
        return self.rate_limiter.get_metrics()
