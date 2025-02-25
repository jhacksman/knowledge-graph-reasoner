"""Rate limiter implementation for API clients."""
import asyncio
import time
from typing import Dict, Optional, Callable, Any, Awaitable
import logging

log = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls.
    
    Implements a token bucket algorithm for rate limiting.
    """
    
    def __init__(
        self,
        calls_per_minute: int = 60,
        burst_limit: int = 10,
        retry_interval: float = 1.0,
        max_retries: int = 3
    ):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
            burst_limit: Maximum number of calls allowed in a burst
            retry_interval: Time to wait between retries in seconds
            max_retries: Maximum number of retries for rate limited requests
        """
        self.calls_per_minute = calls_per_minute
        self.burst_limit = burst_limit
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        
        # Token bucket parameters
        self.tokens: float = float(burst_limit)
        self.token_rate = calls_per_minute / 60.0  # tokens per second
        self.last_refill_time = time.time()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Metrics
        self.metrics: Dict[str, int] = {
            "total_calls": 0,
            "rate_limited_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retried_calls": 0
        }
    
    async def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.token_rate
        
        self.tokens = min(self.burst_limit, self.tokens + new_tokens)
        self.last_refill_time = now
    
    async def _consume_token(self) -> bool:
        """Consume a token if available.
        
        Returns:
            bool: True if token was consumed, False otherwise
        """
        async with self.lock:
            await self._refill_tokens()
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    async def wait_for_token(self, timeout: Optional[float] = None) -> bool:
        """Wait until a token is available.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if token was acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if await self._consume_token():
                return True
            
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            # Wait before trying again
            wait_time = 1.0 / self.token_rate
            await asyncio.sleep(min(wait_time, 1.0))
    
    async def execute(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute a function with rate limiting.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Any: Result of func
            
        Raises:
            Exception: If func raises an exception after max_retries
        """
        self.metrics["total_calls"] += 1
        
        for retry in range(self.max_retries + 1):
            # Wait for token
            if not await self.wait_for_token():
                self.metrics["rate_limited_calls"] += 1
                if retry < self.max_retries:
                    self.metrics["retried_calls"] += 1
                    log.warning(f"Rate limited, retrying in {self.retry_interval}s (attempt {retry+1}/{self.max_retries+1})")
                    await asyncio.sleep(self.retry_interval)
                    continue
                else:
                    self.metrics["failed_calls"] += 1
                    raise Exception("Rate limit exceeded and max retries reached")
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                self.metrics["successful_calls"] += 1
                return result
            except Exception as e:
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    self.metrics["rate_limited_calls"] += 1
                    if retry < self.max_retries:
                        self.metrics["retried_calls"] += 1
                        log.warning(f"Rate limited by API, retrying in {self.retry_interval}s (attempt {retry+1}/{self.max_retries+1})")
                        await asyncio.sleep(self.retry_interval)
                        continue
                
                self.metrics["failed_calls"] += 1
                raise
    
    def get_metrics(self) -> Dict[str, int]:
        """Get rate limiter metrics.
        
        Returns:
            Dict[str, int]: Metrics dictionary
        """
        return self.metrics.copy()
