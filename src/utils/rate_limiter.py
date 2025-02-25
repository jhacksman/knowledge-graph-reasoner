"""Rate limiter utility for API requests."""
import asyncio
from asyncio import Task
import logging
import time
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Callable, Awaitable, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
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

class TokenCounter:
    """Token counter for deepseek-r1-671b model."""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text.
        
        This is a simple approximation. For production use,
        consider using a proper tokenizer for deepseek-r1-671b.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Approximate token count
        """
        # Simple approximation: 4 characters per token
        return len(text) // 4 + 1
    
    @staticmethod
    def count_message_tokens(messages: List[Dict[str, str]]) -> int:
        """Count tokens in chat messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            int: Approximate token count
        """
        total = 0
        for message in messages:
            if "content" in message:
                total += TokenCounter.count_tokens(message["content"])
        return total

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.
        
        Args:
            config: Rate limiter configuration
        """
        self.config = config or RateLimitConfig()
        self.minute_window = 60  # seconds
        self.day_window = 86400  # seconds
        self._setup_storage()
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_size)
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._start_worker()
    
    def _setup_storage(self) -> None:
        """Set up persistent storage."""
        db_path = Path(self.config.storage_path)
        
        # Create directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to SQLite database
        self.conn = sqlite3.connect(str(db_path))
        cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS request_log (
            timestamp REAL,
            endpoint TEXT,
            tokens INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_counter (
            date TEXT PRIMARY KEY,
            count INTEGER
        )
        ''')
        
        self.conn.commit()
    
    def _start_worker(self) -> None:
        """Start background worker for processing queued requests."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self) -> None:
        """Process queued requests."""
        while True:
            try:
                func, args, kwargs, future = await self._request_queue.get()
                try:
                    result = await func(*args, **kwargs)
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                finally:
                    self._request_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing queued request: {e}")
    
    async def _get_minute_requests(self) -> List[Tuple[float, str, int]]:
        """Get requests in the last minute.
        
        Returns:
            List[Tuple[float, str, int]]: List of (timestamp, endpoint, tokens)
        """
        async with self._lock:
            cursor = self.conn.cursor()
            now = time.time()
            minute_ago = now - self.minute_window
            cursor.execute(
                "SELECT timestamp, endpoint, tokens FROM request_log WHERE timestamp > ?",
                (minute_ago,)
            )
            return cursor.fetchall()
    
    async def _get_day_requests(self) -> int:
        """Get request count for today.
        
        Returns:
            int: Request count
        """
        async with self._lock:
            cursor = self.conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "SELECT count FROM daily_counter WHERE date = ?",
                (today,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0
    
    async def _log_request(self, endpoint: str, tokens: int) -> None:
        """Log API request.
        
        Args:
            endpoint: API endpoint
            tokens: Token count
        """
        async with self._lock:
            cursor = self.conn.cursor()
            now = time.time()
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Log request
            cursor.execute(
                "INSERT INTO request_log (timestamp, endpoint, tokens) VALUES (?, ?, ?)",
                (now, endpoint, tokens)
            )
            
            # Update daily counter
            cursor.execute(
                "INSERT INTO daily_counter (date, count) VALUES (?, 1) "
                "ON CONFLICT(date) DO UPDATE SET count = count + 1",
                (today,)
            )
            
            self.conn.commit()
    
    async def _clean_old_logs(self) -> None:
        """Clean old request logs."""
        async with self._lock:
            cursor = self.conn.cursor()
            now = time.time()
            day_ago = now - self.day_window
            
            # Delete logs older than a day
            cursor.execute(
                "DELETE FROM request_log WHERE timestamp < ?",
                (day_ago,)
            )
            
            # Delete daily counters older than 30 days
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            cursor.execute(
                "DELETE FROM daily_counter WHERE date < ?",
                (thirty_days_ago,)
            )
            
            self.conn.commit()
    
    async def _check_rate_limits(self, tokens: int) -> Tuple[bool, str]:
        """Check if rate limits are exceeded.
        
        Args:
            tokens: Token count for the request
            
        Returns:
            Tuple[bool, str]: (is_allowed, reason)
        """
        # Get current usage
        minute_requests = await self._get_minute_requests()
        day_requests = await self._get_day_requests()
        
        # Calculate current rates
        minute_request_count = len(minute_requests)
        minute_token_count = sum(r[2] for r in minute_requests)
        
        # Check rate limits
        if minute_request_count >= self.config.calls_per_minute:
            return False, f"Exceeded {self.config.calls_per_minute} requests per minute"
        
        if day_requests >= self.config.calls_per_day:
            return False, f"Exceeded {self.config.calls_per_day} requests per day"
        
        if minute_token_count + tokens > self.config.tokens_per_minute:
            return False, f"Exceeded {self.config.tokens_per_minute} tokens per minute"
        
        return True, ""
    
    async def wait_if_needed(self, tokens: int) -> None:
        """Wait if rate limits are exceeded.
        
        Args:
            tokens: Token count for the request
        """
        await self._clean_old_logs()
        
        retry_count = 0
        while retry_count < self.config.max_retries:
            is_allowed, reason = await self._check_rate_limits(tokens)
            
            if is_allowed:
                return
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = self.config.retry_interval * (2 ** retry_count)
            jitter = random.uniform(0, self.config.jitter_factor * wait_time)
            wait_time += jitter
            
            logger.warning(f"Rate limit exceeded: {reason}. Waiting {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            retry_count += 1
        
        raise RuntimeError(f"Rate limit exceeded after {self.config.max_retries} retries: {reason}")
    
    async def execute(self, func: Callable[..., Awaitable[Any]], *args, endpoint: str, tokens: int, urgent: bool = True, **kwargs) -> Any:
        """Execute function with rate limiting.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            endpoint: API endpoint
            tokens: Token count
            urgent: Whether the request is urgent
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
        """
        if urgent:
            # For urgent requests, wait if needed and execute immediately
            await self.wait_if_needed(tokens)
            result = await func(*args, **kwargs)
            await self._log_request(endpoint, tokens)
            return result
        else:
            # For non-urgent requests, queue for background processing
            future: asyncio.Future = asyncio.Future()
            
            try:
                await asyncio.wait_for(
                    self._request_queue.put((func, args, kwargs, future)),
                    timeout=self.config.non_urgent_timeout
                )
            except asyncio.TimeoutError:
                raise RuntimeError(f"Queue full, request timed out after {self.config.non_urgent_timeout} seconds")
            
            return await future
    
    async def close(self) -> None:
        """Close rate limiter."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
