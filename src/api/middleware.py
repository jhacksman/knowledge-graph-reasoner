"""Middleware for the API."""
from typing import Callable, Dict, List, Optional, Any, Union
import time
import asyncio
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import uuid
from datetime import datetime, timedelta
import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# Setup logging
logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, reset_at: datetime):
        """Initialize the exception."""
        self.limit = limit
        self.reset_at = reset_at
        self.retry_after = int((reset_at - datetime.utcnow()).total_seconds())
        super().__init__(f"Rate limit exceeded. Retry after {self.retry_after} seconds.")


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        # Store request counts per client
        # {client_id: {window_start_time: count}}
        self.minute_windows: Dict[str, Dict[int, int]] = {}
        self.day_windows: Dict[str, Dict[int, int]] = {}
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(
        self, client_id: str, minute_limit: int, day_limit: int
    ) -> None:
        """Check if the client has exceeded rate limits."""
        now = int(time.time())
        minute_window = now // 60
        day_window = now // 86400
        
        async with self.lock:
            # Initialize client windows if needed
            if client_id not in self.minute_windows:
                self.minute_windows[client_id] = {}
            if client_id not in self.day_windows:
                self.day_windows[client_id] = {}
            
            # Clean up old windows
            self._clean_old_windows(client_id, minute_window, day_window)
            
            # Check minute limit
            minute_count = sum(self.minute_windows[client_id].values())
            if minute_count >= minute_limit:
                reset_at = datetime.fromtimestamp((minute_window + 1) * 60)
                raise RateLimitExceeded(minute_limit, reset_at)
            
            # Check day limit
            day_count = sum(self.day_windows[client_id].values())
            if day_count >= day_limit:
                reset_at = datetime.fromtimestamp((day_window + 1) * 86400)
                raise RateLimitExceeded(day_limit, reset_at)
            
            # Increment counters
            if minute_window in self.minute_windows[client_id]:
                self.minute_windows[client_id][minute_window] += 1
            else:
                self.minute_windows[client_id][minute_window] = 1
            
            if day_window in self.day_windows[client_id]:
                self.day_windows[client_id][day_window] += 1
            else:
                self.day_windows[client_id][day_window] = 1
    
    def _clean_old_windows(
        self, client_id: str, current_minute_window: int, current_day_window: int
    ) -> None:
        """Clean up old time windows."""
        # Keep only the current minute window and the previous one
        self.minute_windows[client_id] = {
            window: count
            for window, count in self.minute_windows[client_id].items()
            if window >= current_minute_window - 1
        }
        
        # Keep only the current day window
        self.day_windows[client_id] = {
            window: count
            for window, count in self.day_windows[client_id].items()
            if window >= current_day_window
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware."""
        super().__init__(app)
        self.rate_limiter = RateLimiter()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Set request ID
        request.state.request.state.request_id = str(uuid.uuid4())
        """Process the request."""
        # Skip rate limiting for certain paths
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
            return await call_next(request)
        
        # Get client ID from API key or IP address
        client_id = self._get_client_id(request)
        
        try:
            # Default limits
            minute_limit = 60
            day_limit = 10000
            
            # Check for custom limits in API key
            api_key = request.headers.get("X-API-Key")
            if api_key and hasattr(request.state, "api_key_data"):
                minute_limit = request.state.api_key_data.rate_limit_per_minute
                day_limit = request.state.api_key_data.rate_limit_per_day
            
            # Check rate limits
            await self.rate_limiter.check_rate_limit(client_id, minute_limit, day_limit)
            
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
            response.headers["X-RateLimit-Limit-Day"] = str(day_limit)
            
            return response
            
        except RateLimitExceeded as e:
            # Return rate limit exceeded response
            content = json.dumps({
                "detail": str(e),
                "error_code": "rate_limit_exceeded",
                "retry_after": e.retry_after
            })
            
            response = Response(
                content=content,
                status_code=429,
                media_type="application/json"
            )
            
            response.headers["Retry-After"] = str(e.retry_after)
            response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
            response.headers["X-RateLimit-Limit-Day"] = str(day_limit)
            
            return response
            
        except RateLimitExceeded as e:
            # Return rate limit exceeded response
            content = json.dumps({
                "detail": str(e),
                "error_code": "rate_limit_exceeded",
                "retry_after": e.retry_after
            })
            
            response = Response(
                content=content,
                status_code=429,
                media_type="application/json"
            )
            
            response.headers["Retry-After"] = str(e.retry_after)
            response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
            response.headers["X-RateLimit-Limit-Day"] = str(day_limit)
            
            return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client ID from request."""
        # Try to get from API key
        api_key = request.headers.get("X-API-Key")
        if api_key and hasattr(request.state, "api_key_data"):
            return request.state.api_key_data.client_id
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Set request ID
        request.state.request.state.request_id = str(uuid.uuid4())
        """Process the request."""
        # Skip rate limiting for certain paths
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
            return await call_next(request)
        
        # Get client ID from API key or IP address
        client_id = self._get_client_id(request)
        
        try:
            # Default limits
            minute_limit = 60
            day_limit = 10000
            
            # Check for custom limits in API key
            api_key = request.headers.get("X-API-Key")
            if api_key and hasattr(request.state, "api_key_data"):
                minute_limit = request.state.api_key_data.rate_limit_per_minute
                day_limit = request.state.api_key_data.rate_limit_per_day
            
            # Check rate limits
            await self.rate_limiter.check_rate_limit(client_id, minute_limit, day_limit)
            
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
            response.headers["X-RateLimit-Limit-Day"] = str(day_limit)
            
            return response
            
        except RateLimitExceeded as e:
            # Return rate limit exceeded response
            content = json.dumps({
                "detail": str(e),
                "error_code": "rate_limit_exceeded",
                "retry_after": e.retry_after
            })
            
            response = Response(
                content=content,
                status_code=429,
                media_type="application/json"
            )
            
            response.headers["Retry-After"] = str(e.retry_after)
            response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
            response.headers["X-RateLimit-Limit-Day"] = str(day_limit)
            
            return response
            
        except Exception as e:
            # Log exception
            logger.exception(
                f"Request failed: method={request.method}, "
                f"path={request.url.path}, error={str(e)}, request.state.request_id={request.state.request_id}"
            )
            
            # Re-raise the exception
            raise


def setup_middleware(app: FastAPI) -> None:
    """Set up middleware for the application."""
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, this should be restricted
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware)
