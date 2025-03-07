"""Rate-limited Venice.ai LLM integration."""
from typing import Dict, Any, List
import numpy as np
import logging
from dataclasses import dataclass

from .llm import VeniceLLM, VeniceLLMConfig
from ..utils.rate_limiter import RateLimiter, RateLimitConfig, TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class RateLimitedVeniceLLMConfig(VeniceLLMConfig):
    """Configuration for rate-limited Venice.ai LLM client."""
    
    calls_per_minute: int = 15
    calls_per_day: int = 10000
    tokens_per_minute: int = 200000
    burst_limit: int = 5
    retry_interval: float = 1.0
    max_retries: int = 5
    jitter_factor: float = 0.1
    storage_path: str = ".rate_limit_storage.db"
    queue_size: int = 100
    non_urgent_timeout: float = 60.0


class RateLimitedVeniceLLM(VeniceLLM):
    """Rate-limited Venice.ai LLM client."""
    
    def __init__(self, config: RateLimitedVeniceLLMConfig):
        """Initialize rate-limited LLM client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        self.config = config
        
        # Create rate limiter
        rate_limit_config = RateLimitConfig(
            calls_per_minute=config.calls_per_minute,
            calls_per_day=config.calls_per_day,
            tokens_per_minute=config.tokens_per_minute,
            burst_limit=config.burst_limit,
            retry_interval=config.retry_interval,
            max_retries=config.max_retries,
            jitter_factor=config.jitter_factor,
            storage_path=config.storage_path,
            queue_size=config.queue_size,
            non_urgent_timeout=config.non_urgent_timeout
        )
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.token_counter = TokenCounter()
    
    async def close(self):
        """Close the client session."""
        await super().close()
        await self.rate_limiter.close()
    
    async def embed_text(self, text: str, urgent: bool = True) -> np.ndarray:
        """Get text embedding with rate limiting.
        
        Args:
            text: Text to embed
            urgent: Whether the request is urgent
            
        Returns:
            np.ndarray: Text embedding
        """
        # Count tokens
        tokens = self.token_counter.count_tokens(text)
        
        # Execute with rate limiting
        return await self.rate_limiter.execute(  # type: ignore  # type: ignore
            super().embed_text,
            text,
            endpoint="embeddings",
            tokens=tokens,
            urgent=urgent
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        urgent: bool = True
    ) -> Dict[str, Any]:
        """Generate text completion with rate limiting.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            urgent: Whether the request is urgent
            
        Returns:
            Dict[str, Any]: Response data
        """
        # Count tokens
        tokens = self.token_counter.count_message_tokens(messages) + max_tokens
        
        # Execute with rate limiting
        return await self.rate_limiter.execute(  # type: ignore  # type: ignore
            super().generate,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint="chat/completions",
            tokens=tokens,
            urgent=urgent
        )
