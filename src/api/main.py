"""FastAPI application setup for the knowledge graph reasoner API."""
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import logging
import os
import time
from datetime import datetime
import uuid
from typing import Dict, Any, Optional

from src.api.middleware import setup_middleware
from src.api.auth import get_api_key, ApiKey
from src.api.routes import (
    concepts,
    relationships,
    queries,
    expansion,
    metrics,
    search,
)
from src.graph.manager import GraphManager
from src.api.models import ErrorResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Knowledge Graph Reasoner API",
    description="API for interacting with the knowledge graph reasoner",
    version="0.1.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
)

# Setup middleware
setup_middleware(app)

# Add exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail=f"An unexpected error occurred: {str(exc)}",
            error_code="internal_server_error",
            path=request.url.path,
            timestamp=datetime.utcnow(),
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code=f"http_{exc.status_code}",
            path=request.url.path,
            timestamp=datetime.utcnow(),
        ).dict(),
        headers=exc.headers,
    )


# Add custom docs routes
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Serve custom Swagger UI."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Knowledge Graph Reasoner API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


# Add dependency injection for GraphManager
@app.middleware("http")
async def add_graph_manager(request: Request, call_next):
    """Add GraphManager to request state."""
    # Initialize GraphManager
    graph_manager = GraphManager()
    
    # Add to request state
    request.state.graph_manager = graph_manager
    
    # Process request
    response = await call_next(request)
    
    return response


# Dependency to get GraphManager from request state
async def get_graph_manager(request: Request) -> GraphManager:
    """Get GraphManager from request state."""
    return request.state.graph_manager


# Add routes
app.include_router(concepts.router)
app.include_router(relationships.router)
app.include_router(queries.router)
app.include_router(expansion.router)
app.include_router(metrics.router)
app.include_router(search.router)


# Add root endpoint
@app.get(
    "/",
    summary="API root",
    description="Get API information",
)
async def root(
    api_key: Optional[ApiKey] = Depends(get_api_key),
) -> Dict[str, Any]:
    """Get API information."""
    return {
        "name": "Knowledge Graph Reasoner API",
        "version": "0.1.0",
        "documentation": "/docs",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Add health check endpoint
@app.get(
    "/health",
    summary="Health check",
    description="Check if the API is healthy",
)
async def health_check() -> Dict[str, Any]:
    """Check if the API is healthy."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0,
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Record start time
    app.state.start_time = time.time()
    
    # Log startup
    logger.info("API starting up")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    # Log shutdown
    logger.info("API shutting down")


# Create routes/__init__.py to make routes package importable
if not os.path.exists("src/api/routes/__init__.py"):
    os.makedirs("src/api/routes", exist_ok=True)
    with open("src/api/routes/__init__.py", "w") as f:
        f.write('"""API routes package."""\n')
