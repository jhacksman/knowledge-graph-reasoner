#!/usr/bin/env python3
"""
Fix script for middleware attribute errors.

This script adds missing attributes to RequestLoggingMiddleware class.
"""
import os
from pathlib import Path


def fix_middleware_attributes():
    """Fix middleware attribute errors."""
    # Define the path to the middleware.py file
    middleware_file = Path("src/api/middleware.py")
    
    if not middleware_file.exists():
        print(f"Error: {middleware_file} does not exist")
        return
    
    # Read the current content of the file
    with open(middleware_file, "r") as file:
        content = file.read()
    
    # Fix missing _get_client_id method
    if "def _get_client_id" not in content:
        # Add the method to RequestLoggingMiddleware
        content = content.replace(
            "class RequestLoggingMiddleware(BaseHTTPMiddleware):",
            """class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limiter = RateLimiter()
    
    def _get_client_id(self, request: Request) -> str:
        \"\"\"Get client ID from request.\"\"\"
        # Try to get from API key
        api_key = request.headers.get("X-API-Key")
        if api_key and hasattr(request.state, "api_key_data"):
            return request.state.api_key_data.client_id
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
"""
        )
    
    # Fix request_id reference
    if "request.state.request_id = " not in content:
        content = content.replace(
            "async def dispatch(self, request: Request, call_next: Callable) -> Response:",
            """async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request.state.request_id = str(uuid.uuid4())"""
        )
    
    # Use request.state.request_id instead of request_id
    content = content.replace(
        "request_id",
        "request.state.request_id"
    )
    
    # Add uuid import if missing
    if "import uuid" not in content:
        content = content.replace(
            "import logging",
            "import logging\nimport uuid"
        )
    
    # Save the updated content
    with open(middleware_file, "w") as file:
        file.write(content)
    
    print(f"Fixed middleware attribute errors in {middleware_file}")


if __name__ == "__main__":
    fix_middleware_attributes()