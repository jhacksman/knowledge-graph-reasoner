"""Tests for rate limiter utility."""
import pytest
import pytest_asyncio
import asyncio
import time
import sqlite3
import os
from unittest.mock import AsyncMock, patch

from src.utils.rate_limiter import RateLimiter, RateLimitConfig, TokenCounter


@pytest.fixture
def test_db_path():
    """Create a temporary database path for testing."""
    db_path = ".test_rate_limit_storage.db"
    
    # Ensure the file doesn't exist
    if os.path.exists(db_path):
        os.remove(db_path)
    
    yield db_path
    
    # Clean up after tests
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def test_config(test_db_path):
    """Create a test rate limit configuration."""
    return RateLimitConfig(
        calls_per_minute=15,
        calls_per_day=10000,
        tokens_per_minute=200000,
        burst_limit=5,
        retry_interval=0.1,  # Short interval for testing
        max_retries=3,
        jitter_factor=0.1,
        storage_path=test_db_path,
        queue_size=10,
        non_urgent_timeout=1.0
    )


@pytest_asyncio.fixture
async def rate_limiter(test_config):
    """Create a rate limiter instance for testing."""
    limiter = RateLimiter(test_config)
    yield limiter
    await limiter.close()


class TestTokenCounter:
    """Tests for TokenCounter."""
    
    def test_count_tokens(self):
        """Test token counting for text."""
        # Test empty string
        assert TokenCounter.count_tokens("") == 1
        
        # Test short text
        assert TokenCounter.count_tokens("Hello") == 2  # 5 chars / 4 + 1 = 2
        
        # Test longer text
        text = "This is a longer text that should have more tokens than the short one."
        assert TokenCounter.count_tokens(text) == len(text) // 4 + 1
    
    def test_count_message_tokens(self):
        """Test token counting for chat messages."""
        # Test empty messages
        assert TokenCounter.count_message_tokens([]) == 0
        
        # Test single message
        messages = [{"role": "user", "content": "Hello"}]
        assert TokenCounter.count_message_tokens(messages) == 2
        
        # Test multiple messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        expected = (
            TokenCounter.count_tokens("Hello") +
            TokenCounter.count_tokens("Hi there!") +
            TokenCounter.count_tokens("How are you?")
        )
        assert TokenCounter.count_message_tokens(messages) == expected
        
        # Test message without content
        messages = [{"role": "user"}]
        assert TokenCounter.count_message_tokens(messages) == 0


@pytest.mark.asyncio
class TestRateLimiter:
    """Tests for RateLimiter."""
    
    async def test_init(self, test_config):
        """Test initialization."""
        limiter = RateLimiter(test_config)
        
        # Check that database was created
        assert os.path.exists(test_config.storage_path)
        
        # Check that tables were created
        conn = sqlite3.connect(test_config.storage_path)
        cursor = conn.cursor()
        
        # Check request_log table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='request_log'")
        assert cursor.fetchone() is not None
        
        # Check daily_counter table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_counter'")
        assert cursor.fetchone() is not None
        
        conn.close()
        await limiter.close()
    
    async def test_log_request(self, rate_limiter):
        """Test request logging."""
        # Log a request
        await rate_limiter._log_request("test_endpoint", 100)
        
        # Check that request was logged
        conn = sqlite3.connect(rate_limiter.config.storage_path)
        cursor = conn.cursor()
        
        # Check request_log
        cursor.execute("SELECT COUNT(*) FROM request_log")
        assert cursor.fetchone()[0] == 1
        
        # Check daily_counter
        today = time.strftime("%Y-%m-%d")
        cursor.execute("SELECT count FROM daily_counter WHERE date = ?", (today,))
        assert cursor.fetchone()[0] == 1
        
        conn.close()
    
    async def test_get_minute_requests(self, rate_limiter):
        """Test getting requests in the last minute."""
        # Log some requests
        await rate_limiter._log_request("test_endpoint1", 100)
        await rate_limiter._log_request("test_endpoint2", 200)
        
        # Get minute requests
        requests = await rate_limiter._get_minute_requests()
        
        # Check results
        assert len(requests) == 2
        assert requests[0][1] == "test_endpoint1"
        assert requests[0][2] == 100
        assert requests[1][1] == "test_endpoint2"
        assert requests[1][2] == 200
    
    async def test_get_day_requests(self, rate_limiter):
        """Test getting request count for today."""
        # Log some requests
        await rate_limiter._log_request("test_endpoint1", 100)
        await rate_limiter._log_request("test_endpoint2", 200)
        
        # Get day requests
        count = await rate_limiter._get_day_requests()
        
        # Check results
        assert count == 2
    
    async def test_clean_old_logs(self, rate_limiter):
        """Test cleaning old request logs."""
        # Connect to database
        conn = sqlite3.connect(rate_limiter.config.storage_path)
        cursor = conn.cursor()
        
        # Insert old log
        old_time = time.time() - 86400 * 2  # 2 days ago
        cursor.execute(
            "INSERT INTO request_log (timestamp, endpoint, tokens) VALUES (?, ?, ?)",
            (old_time, "old_endpoint", 100)
        )
        
        # Insert old daily counter
        old_date = (time.strftime("%Y-%m-%d", time.localtime(old_time - 86400 * 30)))  # 30 days ago
        cursor.execute(
            "INSERT INTO daily_counter (date, count) VALUES (?, ?)",
            (old_date, 10)
        )
        
        conn.commit()
        conn.close()
        
        # Clean old logs
        await rate_limiter._clean_old_logs()
        
        # Check that old logs were deleted
        conn = sqlite3.connect(rate_limiter.config.storage_path)
        cursor = conn.cursor()
        
        # Check request_log
        cursor.execute("SELECT COUNT(*) FROM request_log WHERE timestamp < ?", (time.time() - 86400,))
        assert cursor.fetchone()[0] == 0
        
        # Check daily_counter
        thirty_days_ago = (time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400 * 30)))
        cursor.execute("SELECT COUNT(*) FROM daily_counter WHERE date < ?", (thirty_days_ago,))
        assert cursor.fetchone()[0] == 0
        
        conn.close()
    
    async def test_check_rate_limits(self, rate_limiter):
        """Test checking rate limits."""
        # Test with no requests
        is_allowed, _ = await rate_limiter._check_rate_limits(100)
        assert is_allowed is True
        
        # Test with requests under limits
        for i in range(10):
            await rate_limiter._log_request(f"test_endpoint{i}", 10000)
        
        is_allowed, _ = await rate_limiter._check_rate_limits(100)
        assert is_allowed is True
        
        # Test with requests over minute limit
        for i in range(5):
            await rate_limiter._log_request(f"test_endpoint{i+10}", 10000)
        
        is_allowed, reason = await rate_limiter._check_rate_limits(100)
        assert is_allowed is False
        assert "requests per minute" in reason
        
        # Test with requests over token limit
        # First, clear the database
        conn = sqlite3.connect(rate_limiter.config.storage_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM request_log")
        conn.commit()
        conn.close()
        
        # Log requests with high token count
        await rate_limiter._log_request("test_endpoint", 190000)
        
        is_allowed, reason = await rate_limiter._check_rate_limits(20000)
        assert is_allowed is False
        assert "tokens per minute" in reason
    
    async def test_wait_if_needed(self, rate_limiter):
        """Test waiting if rate limits are exceeded."""
        # Mock _check_rate_limits to simulate rate limit exceeded
        original_check = rate_limiter._check_rate_limits
        check_calls = 0
        
        async def mock_check_rate_limits(tokens):
            nonlocal check_calls
            check_calls += 1
            if check_calls == 1:
                return False, "Test rate limit exceeded"
            return True, ""
        
        rate_limiter._check_rate_limits = mock_check_rate_limits
        
        # Test waiting
        await rate_limiter.wait_if_needed(100)
        elapsed = time.time() - start_time
        
        # Should have waited at least retry_interval
        assert elapsed >= rate_limiter.config.retry_interval
        assert check_calls == 2
        
        # Restore original method
        rate_limiter._check_rate_limits = original_check
    
    async def test_execute_urgent(self, rate_limiter):
        """Test executing urgent requests."""
        # Create mock function
        mock_func = AsyncMock()
        mock_func.return_value = "test_result"
        
        # Execute function
        result = await rate_limiter.execute(
            mock_func, "arg1", "arg2", 
            endpoint="test_endpoint", 
            tokens=100,
            urgent=True,
            kwarg1="value1"
        )
        
        # Check results
        assert result == "test_result"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
        
        # Check that request was logged
        conn = sqlite3.connect(rate_limiter.config.storage_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM request_log WHERE endpoint = ?", ("test_endpoint",))
        assert cursor.fetchone()[0] == 1
        conn.close()
    
    async def test_execute_non_urgent(self, rate_limiter):
        """Test executing non-urgent requests."""
        # Create mock function
        mock_func = AsyncMock()
        mock_func.return_value = "test_result"
        
        # Execute function
        task = asyncio.create_task(
            rate_limiter.execute(
                mock_func, "arg1", "arg2", 
                endpoint="test_endpoint", 
                tokens=100,
                urgent=False,
                kwarg1="value1"
            )
        )
        
        # Wait for task to complete
        result = await task
        
        # Check results
        assert result == "test_result"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
    
    async def test_execute_rate_limited(self, rate_limiter):
        """Test executing rate-limited requests."""
        # Mock wait_if_needed to simulate rate limiting
        original_wait = rate_limiter.wait_if_needed
        wait_calls = 0
        
        async def mock_wait_if_needed(tokens):
            nonlocal wait_calls
            wait_calls += 1
            await asyncio.sleep(0.1)  # Simulate waiting
        
        rate_limiter.wait_if_needed = mock_wait_if_needed
        
        # Create mock function
        mock_func = AsyncMock()
        mock_func.return_value = "test_result"
        
        # Execute function
        result = await rate_limiter.execute(
            mock_func, "arg1", "arg2", 
            endpoint="test_endpoint", 
            tokens=100,
            urgent=True,
            kwarg1="value1"
        )
        
        # Check results
        assert result == "test_result"
        assert wait_calls == 1
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
        
        # Restore original method
        rate_limiter.wait_if_needed = original_wait
    
    async def test_execute_error_handling(self, rate_limiter):
        """Test error handling in execute."""
        # Create mock function that raises an exception
        mock_func = AsyncMock()
        mock_func.side_effect = ValueError("Test error")
        
        # Execute function and check that exception is propagated
        with pytest.raises(ValueError, match="Test error"):
            await rate_limiter.execute(
                mock_func, "arg1", "arg2", 
                endpoint="test_endpoint", 
                tokens=100,
                urgent=True,
                kwarg1="value1"
            )
    
    async def test_queue_full(self, rate_limiter):
        """Test behavior when queue is full."""
        # Create a rate limiter with a small queue
        config = RateLimitConfig(
            calls_per_minute=15,
            calls_per_day=10000,
            tokens_per_minute=200000,
            queue_size=1,
            non_urgent_timeout=0.1
        )
        limiter = RateLimiter(config)
        
        # Create a slow mock function
        async def slow_func(*args, **kwargs):
            await asyncio.sleep(0.5)
            return "result"
        
        # Fill the queue
        task1 = asyncio.create_task(
            limiter.execute(
                slow_func,
                endpoint="test_endpoint",
                tokens=100,
                urgent=False
            )
        )
        
        # Wait a bit to ensure the queue is filled
        await asyncio.sleep(0.1)
        
        # Try to add another task and check that it times out
        with pytest.raises(RuntimeError, match="Queue full"):
            await limiter.execute(
                slow_func,
                endpoint="test_endpoint",
                tokens=100,
                urgent=False
            )
        
        # Clean up
        await task1
        await limiter.close()
    
    async def test_close(self, test_config):
        """Test closing the rate limiter."""
        limiter = RateLimiter(test_config)
        
        # Spy on worker task
        original_task = limiter._worker_task
        
        # Close limiter
        await limiter.close()
        
        # Check that worker task was cancelled
        assert original_task.cancelled()
        
        # Check that database connection was closed
        with pytest.raises(sqlite3.ProgrammingError):
            limiter.conn.execute("SELECT 1")


@pytest.mark.asyncio
class TestRateLimiterIntegration:
    """Integration tests for RateLimiter."""
    
    async def test_rate_limiting_requests_per_minute(self, test_config):
        """Test rate limiting for requests per minute."""
        # Create a rate limiter with a low requests per minute limit
        config = RateLimitConfig(
            calls_per_minute=5,  # Low limit for testing
            calls_per_day=10000,
            tokens_per_minute=200000,
            retry_interval=0.1,
            max_retries=2,
            storage_path=test_config.storage_path
        )
        limiter = RateLimiter(config)
        
        # Create a mock function
        mock_func = AsyncMock()
        mock_func.return_value = "test_result"
        
        # Execute function multiple times
        for i in range(5):
            await limiter.execute(
                mock_func,
                endpoint="test_endpoint",
                tokens=100,
                urgent=True
            )
        
        # The next call should hit the rate limit and wait
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            try:
                await limiter.execute(
                    mock_func,
                    endpoint="test_endpoint",
                    tokens=100,
                    urgent=True
                )
            except RuntimeError:
                # This is expected if the rate limit is hit and max retries are exceeded
                pass
        
        # Check that sleep was called (indicating rate limiting)
        assert mock_sleep.called
        
        # Clean up
        await limiter.close()
    
    async def test_rate_limiting_tokens_per_minute(self, test_config):
        """Test rate limiting for tokens per minute."""
        # Create a rate limiter with a low tokens per minute limit
        config = RateLimitConfig(
            calls_per_minute=100,
            calls_per_day=10000,
            tokens_per_minute=500,  # Low limit for testing
            retry_interval=0.1,
            max_retries=2,
            storage_path=test_config.storage_path
        )
        limiter = RateLimiter(config)
        
        # Create a mock function
        mock_func = AsyncMock()
        mock_func.return_value = "test_result"
        
        # Execute function with high token count
        await limiter.execute(
            mock_func,
            endpoint="test_endpoint",
            tokens=400,
            urgent=True
        )
        
        # The next call should hit the rate limit and wait
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            try:
                await limiter.execute(
                    mock_func,
                    endpoint="test_endpoint",
                    tokens=200,
                    urgent=True
                )
            except RuntimeError:
                # This is expected if the rate limit is hit and max retries are exceeded
                pass
        
        # Check that sleep was called (indicating rate limiting)
        assert mock_sleep.called
        
        # Clean up
        await limiter.close()
    
    async def test_rate_limiting_calls_per_day(self, test_config):
        """Test rate limiting for calls per day."""
        # Create a rate limiter with a low calls per day limit
        config = RateLimitConfig(
            calls_per_minute=100,
            calls_per_day=3,  # Low limit for testing
            tokens_per_minute=200000,
            retry_interval=0.1,
            max_retries=2,
            storage_path=test_config.storage_path
        )
        limiter = RateLimiter(config)
        
        # Create a mock function
        mock_func = AsyncMock()
        mock_func.return_value = "test_result"
        
        # Execute function multiple times
        for i in range(3):
            await limiter.execute(
                mock_func,
                endpoint="test_endpoint",
                tokens=100,
                urgent=True
            )
        
        # The next call should hit the rate limit and wait
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            try:
                await limiter.execute(
                    mock_func,
                    endpoint="test_endpoint",
                    tokens=100,
                    urgent=True
                )
            except RuntimeError:
                # This is expected if the rate limit is hit and max retries are exceeded
                pass
        
        # Check that sleep was called (indicating rate limiting)
        assert mock_sleep.called
        
        # Clean up
        await limiter.close()
