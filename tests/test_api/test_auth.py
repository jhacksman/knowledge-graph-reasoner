"""Tests for the API authentication and authorization."""
import pytest
from fastapi import HTTPException, Request
from datetime import datetime, timedelta
from src.api.auth import (
    get_api_key, has_permission, Permission, ApiKey,
    Role, ROLE_PERMISSIONS
)


class MockRequest:
    """Mock request for testing."""
    
    def __init__(self, headers=None):
        """Initialize mock request."""
        self.headers = headers or {}
        self.state = type('obj', (object,), {})  # Create a simple object with state


def test_api_key_model():
    """Test ApiKey model."""
    created_time = datetime.utcnow()
    api_key = ApiKey(
        key="test-key",
        client_id="test-client",
        role=Role.ADMIN,
        created_at=created_time,
        expires_at=created_time + timedelta(days=7),
        rate_limit_per_minute=60,
        rate_limit_per_day=10000,
        last_used=None,
        usage_count=0,
    )
    assert api_key.key == "test-key"
    assert api_key.client_id == "test-client"
    assert api_key.role == Role.ADMIN
    assert api_key.created_at == created_time
    assert api_key.last_used is None
    assert api_key.usage_count == 0


def test_permission_enum():
    """Test Permission enum."""
    assert Permission.READ_CONCEPTS.value == "read:concepts"
    assert Permission.WRITE_CONCEPTS.value == "write:concepts"
    assert Permission.READ_RELATIONSHIPS.value == "read:relationships"
    assert Permission.WRITE_RELATIONSHIPS.value == "write:relationships"
    assert Permission.EXECUTE_QUERIES.value == "execute:queries"
    assert Permission.MANAGE_EXPANSION.value == "manage:expansion"
    assert Permission.READ_METRICS.value == "read:metrics"
    assert Permission.READ_PERMISSION_CONCEPTS.value == "read_permission:concepts"
    assert Permission.WRITE_PERMISSION_CONCEPTS.value == "write_permission:concepts"
    assert Permission.READ_PERMISSION_RELATIONSHIPS.value == "read_permission:relationships"
    assert Permission.WRITE_PERMISSION_RELATIONSHIPS.value == "write_permission:relationships"
    assert Permission.READ_PERMISSION_QUERIES.value == "read_permission:queries"
    assert Permission.WRITE_PERMISSION_QUERIES.value == "write_permission:queries"
    assert Permission.READ_PERMISSION_METRICS.value == "read_permission:metrics"
    assert Permission.WRITE_PERMISSION.value == "write:permission"
    assert Permission.ADMIN_PERMISSION_ACCESS.value == "admin_permission:access"


@pytest.mark.asyncio
async def test_get_api_key_missing():
    """Test get_api_key with missing key."""
    # Test with missing API key
    request = MockRequest()
    
    # Since we're testing an async function, we need to await it
    with pytest.raises(HTTPException) as excinfo:
        await get_api_key("", "")  # Pass empty strings for header and query
    
    assert excinfo.value.status_code == 401
    assert "API key is missing" in excinfo.value.detail


@pytest.mark.asyncio
async def test_has_permission_admin():
    """Test has_permission with admin role."""
    # Mock an admin API key
    created_time = datetime.utcnow()
    api_key = ApiKey(
        key="admin-key",
        client_id="admin-client",
        role=Role.ADMIN,
        created_at=created_time,
        rate_limit_per_minute=60,
        rate_limit_per_day=10000,
    )
    
    # Admin should have all permissions
    for permission in Permission:
        dependency = has_permission(permission.value)
        # Since the dependency returns an async function, we need to await it
        result = await dependency(api_key)
        assert result is True  # Permission granted should return True


@pytest.mark.asyncio
async def test_has_permission_contributor():
    """Test has_permission with contributor role."""
    # Mock a contributor API key
    created_time = datetime.utcnow()
    api_key = ApiKey(
        key="contributor-key",
        client_id="contributor-client",
        role=Role.CONTRIBUTOR,
        created_at=created_time,
        rate_limit_per_minute=60,
        rate_limit_per_day=10000,
    )
    
    # Check permissions that contributor should have
    contributor_permissions = ROLE_PERMISSIONS[Role.CONTRIBUTOR]
    for permission_value in contributor_permissions:
        dependency = has_permission(permission_value)
        result = await dependency(api_key)
        assert result is True
    
    # Check permissions that contributor should not have
    admin_only_permissions = [
        Permission.MANAGE_EXPANSION.value,
        Permission.ADMIN_ACCESS.value,
        Permission.ADMIN_PERMISSION_ACCESS.value
    ]
    
    for permission_value in admin_only_permissions:
        dependency = has_permission(permission_value)
        with pytest.raises(HTTPException) as excinfo:
            await dependency(api_key)
        assert excinfo.value.status_code == 403
        assert "Permission denied" in excinfo.value.detail


@pytest.mark.asyncio
async def test_has_permission_readonly():
    """Test has_permission with readonly role."""
    # Mock a readonly API key
    created_time = datetime.utcnow()
    api_key = ApiKey(
        key="readonly-key",
        client_id="readonly-client",
        role=Role.READ_ONLY,
        created_at=created_time,
        rate_limit_per_minute=60,
        rate_limit_per_day=10000,
    )
    
    # Check permissions that readonly should have
    readonly_permissions = ROLE_PERMISSIONS[Role.READ_ONLY]
    for permission_value in readonly_permissions:
        dependency = has_permission(permission_value)
        result = await dependency(api_key)
        assert result is True
    
    # Check permissions that readonly should not have
    write_permissions = [
        Permission.WRITE_CONCEPTS.value,
        Permission.WRITE_RELATIONSHIPS.value,
        Permission.WRITE_QUERIES.value,
        Permission.EXECUTE_QUERIES.value,
        Permission.MANAGE_EXPANSION.value,
        Permission.ADMIN_ACCESS.value,
        Permission.WRITE_PERMISSION_CONCEPTS.value,
        Permission.WRITE_PERMISSION_RELATIONSHIPS.value,
        Permission.WRITE_PERMISSION_QUERIES.value,
        Permission.WRITE_PERMISSION.value,
        Permission.ADMIN_PERMISSION_ACCESS.value
    ]
    
    for permission_value in write_permissions:
        dependency = has_permission(permission_value)
        with pytest.raises(HTTPException) as excinfo:
            await dependency(api_key)
        assert excinfo.value.status_code == 403
        assert "Permission denied" in excinfo.value.detail
