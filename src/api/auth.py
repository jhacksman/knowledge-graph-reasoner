"""Authentication and authorization for the API."""
from typing import Dict, List, Optional, Callable, Any
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import time
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
import json
import uuid

# Setup logging
logger = logging.getLogger(__name__)

# API key security schemes
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)

# Role definitions
class Role:
    """Role definitions for access control."""
    
    READ_ONLY = "read_only"
    CONTRIBUTOR = "contributor"
    ADMIN = "admin"


class Permission:
    """Permission definitions for access control."""
    
    READ_CONCEPTS = "read:concepts"
    WRITE_CONCEPTS = "write:concepts"
    READ_RELATIONSHIPS = "read:relationships"
    WRITE_RELATIONSHIPS = "write:relationships"
    READ_QUERIES = "read:queries"
    WRITE_QUERIES = "write:queries"
    MANAGE_EXPANSION = "manage:expansion"
    READ_METRICS = "read:metrics"
    ADMIN_ACCESS = "admin:access"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    Role.READ_ONLY: [
        Permission.READ_PERMISSION_CONCEPTS,
        Permission.READ_PERMISSION_RELATIONSHIPS,
        Permission.READ_PERMISSION_QUERIES,
        Permission.READ_PERMISSION_METRICS,
    ],
    Role.CONTRIBUTOR: [
        Permission.READ_PERMISSION_CONCEPTS,
        Permission.WRITE_PERMISSION_CONCEPTS,
        Permission.READ_PERMISSION_RELATIONSHIPS,
        Permission.WRITE_PERMISSION_RELATIONSHIPS,
        Permission.READ_PERMISSION_QUERIES,
        Permission.WRITE_PERMISSION_QUERIES,
        Permission.READ_PERMISSION_METRICS,
    ],
    Role.ADMIN: [
        Permission.READ_PERMISSION_CONCEPTS,
        Permission.WRITE_PERMISSION_CONCEPTS,
        Permission.READ_PERMISSION_RELATIONSHIPS,
        Permission.WRITE_PERMISSION_RELATIONSHIPS,
        Permission.READ_PERMISSION_QUERIES,
        Permission.WRITE_PERMISSION_QUERIES,
        Permission.MANAGE_EXPANSION,
        Permission.READ_PERMISSION_METRICS,
        Permission.ADMIN_PERMISSION_ACCESS,
    ],
}


class ApiKey(BaseModel):
    """API key model with associated metadata."""
    
    key: str
    client_id: str
    role: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000
    last_used: Optional[datetime] = None
    usage_count: int = 0


# In-memory API key store (would be replaced with a database in production)
API_KEYS: Dict[str, ApiKey] = {}


def initialize_default_keys():
    """Initialize default API keys for development."""
    # This would be replaced with loading from a secure storage in production
    if not API_KEYS:
        API_KEYS["dev-read-only"] = ApiKey(
            key="dev-read-only",
            client_id="dev-client",
            role=Role.READ_ONLY,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=30),
            rate_limit_per_minute=100,
            rate_limit_per_day=10000,
        )
        
        API_KEYS["dev-contributor"] = ApiKey(
            key="dev-contributor",
            client_id="dev-client",
            role=Role.CONTRIBUTOR,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=30),
            rate_limit_per_minute=100,
            rate_limit_per_day=10000,
        )
        
        API_KEYS["dev-admin"] = ApiKey(
            key="dev-admin",
            client_id="dev-client",
            role=Role.ADMIN,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=30),
            rate_limit_per_minute=100,
            rate_limit_per_day=10000,
        )


# Initialize default keys
initialize_default_keys()


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
) -> ApiKey:
    """Get and validate API key from header or query parameter."""
    api_key = api_key_header or api_key_query
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    key_data = API_KEYS[api_key]
    
    # Check if key has expired
    if key_data.expires_at and key_data.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Update usage statistics
    key_data.last_used = datetime.utcnow()
    key_data.usage_count += 1
    
    return key_data


def has_permission(permission: str):
    """Dependency to check if the user has a specific permission."""
    
    async def check_permission(api_key: ApiKey = Depends(get_api_key)) -> bool:
        """Check if the API key has the required permission."""
        if permission in ROLE_PERMISSIONS.get(api_key.role, []):
            return True
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission} is required",
        )
    
    return check_permission


def verify_request_signature(request_data: Dict[str, Any], signature: str, secret: str) -> bool:
    """Verify the signature of a request."""
    # Sort the request data by key to ensure consistent ordering
    sorted_data = json.dumps(request_data, sort_keys=True)
    
    # Create HMAC signature
    expected_signature = hmac.new(
        secret.encode(),
        sorted_data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures using constant-time comparison
    return hmac.compare_digest(expected_signature, signature)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def track_client_usage(api_key: ApiKey, endpoint: str, request_id: str):
    """Track client usage for analytics and rate limiting."""
    # In a production system, this would store usage data in a database
    logger.info(
        f"API request: client_id={api_key.client_id}, "
        f"endpoint={endpoint}, request_id={request_id}"
    )
