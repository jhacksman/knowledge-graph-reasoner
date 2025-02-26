#!/usr/bin/env python3
"""
Fix script for return type errors.

This script fixes functions returning Any instead of specific types.
"""
import os
import re
from pathlib import Path


def fix_middleware_return_types():
    """Fix return type errors in middleware.py."""
    middleware_file = Path("src/api/middleware.py")
    
    if not middleware_file.exists():
        print(f"Error: {middleware_file} does not exist")
        return
    
    # Read the current content of the file
    with open(middleware_file, "r") as file:
        content = file.read()
    
    # Fix request_id reference
    content = content.replace(
        "logger.info(f\"Request: {request.method} {request.url.path} - {request_id}\")",
        "logger.info(f\"Request: {request.method} {request.url.path} - {request.state.request_id}\")"
    )
    
    # Fix _get_client_id method implementation
    if "_get_client_id" not in content:
        # Add the method to RequestLoggingMiddleware
        content = content.replace(
            "class RequestLoggingMiddleware(BaseHTTPMiddleware):",
            """class RequestLoggingMiddleware(BaseHTTPMiddleware):
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
    
    # Fix rate_limiter reference by adding initialization if missing
    if "self.rate_limiter" not in content:
        content = content.replace(
            "    async def dispatch(self, request: Request, call_next: Callable) -> Response:",
            """    def __init__(self, app: ASGIApp):
        \"\"\"Initialize middleware.\"\"\"
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:"""
        )
    
    # Ensure request_id is set in the request state
    if "request.state.request_id = " not in content:
        content = content.replace(
            "    async def dispatch(self, request: Request, call_next: Callable) -> Response:",
            """    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Set request ID
        request.state.request_id = str(uuid.uuid4())"""
        )
    
    # Add uuid import if not present
    if "import uuid" not in content:
        content = content.replace(
            "import logging",
            "import logging\nimport uuid"
        )
    
    # Save the updated content
    with open(middleware_file, "w") as file:
        file.write(content)
    
    print(f"Fixed return type errors in {middleware_file}")


def fix_metrics_return_types():
    """Fix return type errors in metrics.py."""
    metrics_file = Path("src/metrics/metrics.py")
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} does not exist")
        return
    
    # Read the current content of the file
    with open(metrics_file, "r") as file:
        content = file.read()
    
    # Fix return types by explicitly casting return values
    content = re.sub(
        r"(\s+)return (\w+)",
        r"\1return float(\2)",
        content
    )
    
    # Fix dict return types
    content = re.sub(
        r"(\s+)return (\{.*\})",
        r"\1return dict(\2)",
        content
    )
    
    # Save the updated content
    with open(metrics_file, "w") as file:
        file.write(content)
    
    print(f"Fixed return type errors in {metrics_file}")


def fix_llm_return_types():
    """Fix return type errors in llm.py."""
    llm_file = Path("src/reasoning/llm.py")
    
    if not llm_file.exists():
        print(f"Error: {llm_file} does not exist")
        return
    
    # Read the current content of the file
    with open(llm_file, "r") as file:
        content = file.read()
    
    # Fix dict return types by explicitly casting
    content = re.sub(
        r"(\s+)return await response\.json\(\)",
        r"\1data = await response.json()\n\1return dict(data)",
        content
    )
    
    # Save the updated content
    with open(llm_file, "w") as file:
        file.write(content)
    
    print(f"Fixed return type errors in {llm_file}")


def fix_milvus_store_return_types():
    """Fix return type errors in milvus_store.py."""
    milvus_file = Path("src/vector_store/milvus_store.py")
    
    if not milvus_file.exists():
        print(f"Error: {milvus_file} does not exist")
        return
    
    # Read the current content of the file
    with open(milvus_file, "r") as file:
        content = file.read()
    
    # Fix get_all_nodes and get_all_edges return type hints
    content = content.replace(
        "def get_all_nodes(self) -> AsyncIterator[Node]:",
        "async def get_all_nodes(self) -> AsyncIterator[Node]:"
    )
    
    content = content.replace(
        "def get_all_edges(self) -> AsyncIterator[Edge]:",
        "async def get_all_edges(self) -> AsyncIterator[Edge]:"
    )
    
    # Save the updated content
    with open(milvus_file, "w") as file:
        file.write(content)
    
    print(f"Fixed return type errors in {milvus_file}")


def fix_all_return_types():
    """Fix all return type errors."""
    fix_middleware_return_types()
    fix_metrics_return_types()
    fix_llm_return_types()
    fix_milvus_store_return_types()


if __name__ == "__main__":
    fix_all_return_types()