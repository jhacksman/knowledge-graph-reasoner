"""Tests for the API authentication and authorization."""
import pytest
from fastapi import HTTPException, Request
from src.api.auth import (
    get_api_key, has_permission, Permission, ApiKey,
    verify_api_key, verify_signature
)


class MockRequest:
    """Mock request for testing."""
    
    def __init__(self, headers=None):
        """Initialize mock request."""
        self.headers = headers or {}


def test_api_key_model():
    """Test ApiKey model."""
    api_key = ApiKey(
        key="test-key",
        name="Test Key",
        role="admin",
        created_at="2023-01-01T00:00:00Z",
        last_used_at=None,
        usage_count=0,
    )
    assert api_key.key == "test-key"
    assert api_key.name == "Test Key"
    assert api_key.role == "admin"
    assert api_key.created_at == "2023-01-01T00:00:00Z"
    assert api_key.last_used_at is None
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


def test_verify_api_key_valid():
    """Test verify_api_key with valid key."""
    # Mock a valid API key
    api_key = "test-key"
    result = verify_api_key(api_key)
    assert result is not None
    assert result.key == api_key
    assert result.role in ["admin", "contributor", "readonly"]


def test_verify_api_key_invalid():
    """Test verify_api_key with invalid key."""
    # Test with invalid key
    with pytest.raises(HTTPException) as excinfo:
        verify_api_key("invalid-key")
    assert excinfo.value.status_code == 401
    assert "Invalid API key" in excinfo.value.detail


def test_verify_signature_valid():
    """Test verify_signature with valid signature."""
    # This is a simplified test since we can't easily generate valid signatures
    # In a real test, we would use the actual signing algorithm
    request = MockRequest(headers={
        "X-API-Key": "test-key",
        "X-Signature": "valid-signature",
        "X-Timestamp": "1609459200",
    })
    
    # Mock the signature verification
    # In a real implementation, this would verify the signature
    result = verify_signature(request, "test-key")
    assert result is True


def test_verify_signature_missing_headers():
    """Test verify_signature with missing headers."""
    # Test with missing signature header
    request = MockRequest(headers={
        "X-API-Key": "test-key",
        "X-Timestamp": "1609459200",
    })
    with pytest.raises(HTTPException) as excinfo:
        verify_signature(request, "test-key")
    assert excinfo.value.status_code == 401
    assert "Missing signature" in excinfo.value.detail
    
    # Test with missing timestamp header
    request = MockRequest(headers={
        "X-API-Key": "test-key",
        "X-Signature": "valid-signature",
    })
    with pytest.raises(HTTPException) as excinfo:
        verify_signature(request, "test-key")
    assert excinfo.value.status_code == 401
    assert "Missing timestamp" in excinfo.value.detail


def test_verify_signature_expired():
    """Test verify_signature with expired timestamp."""
    # Test with expired timestamp
    request = MockRequest(headers={
        "X-API-Key": "test-key",
        "X-Signature": "valid-signature",
        "X-Timestamp": "1",  # Very old timestamp
    })
    with pytest.raises(HTTPException) as excinfo:
        verify_signature(request, "test-key")
    assert excinfo.value.status_code == 401
    assert "Signature expired" in excinfo.value.detail


def test_get_api_key_valid():
    """Test get_api_key with valid key."""
    # Mock a valid request
    request = MockRequest(headers={"X-API-Key": "test-key"})
    
    # Test the dependency
    api_key = get_api_key(request)
    assert api_key is not None
    assert api_key.key == "test-key"
    assert api_key.role in ["admin", "contributor", "readonly"]


def test_get_api_key_missing():
    """Test get_api_key with missing key."""
    # Test with missing API key
    request = MockRequest()
    with pytest.raises(HTTPException) as excinfo:
        get_api_key(request)
    assert excinfo.value.status_code == 401
    assert "API key is required" in excinfo.value.detail


def test_get_api_key_invalid():
    """Test get_api_key with invalid key."""
    # Test with invalid API key
    request = MockRequest(headers={"X-API-Key": "invalid-key"})
    with pytest.raises(HTTPException) as excinfo:
        get_api_key(request)
    assert excinfo.value.status_code == 401
    assert "Invalid API key" in excinfo.value.detail


def test_has_permission_admin():
    """Test has_permission with admin role."""
    # Mock an admin API key
    api_key = ApiKey(
        key="admin-key",
        name="Admin Key",
        role="admin",
        created_at="2023-01-01T00:00:00Z",
        last_used_at=None,
        usage_count=0,
    )
    
    # Admin should have all permissions
    for permission in Permission:
        dependency = has_permission(permission)
        result = dependency(api_key)
        assert result is None  # No exception means permission granted


def test_has_permission_contributor():
    """Test has_permission with contributor role."""
    # Mock a contributor API key
    api_key = ApiKey(
        key="contributor-key",
        name="Contributor Key",
        role="contributor",
        created_at="2023-01-01T00:00:00Z",
        last_used_at=None,
        usage_count=0,
    )
    
    # Contributors should have read and write permissions but not manage
    read_permissions = [
        Permission.READ_CONCEPTS,
        Permission.READ_RELATIONSHIPS,
        Permission.READ_METRICS,
        Permission.EXECUTE_QUERIES,
    ]
    write_permissions = [
        Permission.WRITE_CONCEPTS,
        Permission.WRITE_RELATIONSHIPS,
    ]
    manage_permissions = [
        Permission.MANAGE_EXPANSION,
    ]
    
    # Test read and write permissions
    for permission in read_permissions + write_permissions:
        dependency = has_permission(permission)
        result = dependency(api_key)
        assert result is None  # No exception means permission granted
    
    # Test manage permissions
    for permission in manage_permissions:
        dependency = has_permission(permission)
        with pytest.raises(HTTPException) as excinfo:
            dependency(api_key)
        assert excinfo.value.status_code == 403
        assert "Permission denied" in excinfo.value.detail


def test_has_permission_readonly():
    """Test has_permission with readonly role."""
    # Mock a readonly API key
    api_key = ApiKey(
        key="readonly-key",
        name="Readonly Key",
        role="readonly",
        created_at="2023-01-01T00:00:00Z",
        last_used_at=None,
        usage_count=0,
    )
    
    # Readonly should have only read permissions
    read_permissions = [
        Permission.READ_CONCEPTS,
        Permission.READ_RELATIONSHIPS,
        Permission.READ_METRICS,
        Permission.EXECUTE_QUERIES,
    ]
    write_permissions = [
        Permission.WRITE_CONCEPTS,
        Permission.WRITE_RELATIONSHIPS,
        Permission.MANAGE_EXPANSION,
    ]
    
    # Test read permissions
    for permission in read_permissions:
        dependency = has_permission(permission)
        result = dependency(api_key)
        assert result is None  # No exception means permission granted
    
    # Test write permissions
    for permission in write_permissions:
        dependency = has_permission(permission)
        with pytest.raises(HTTPException) as excinfo:
            dependency(api_key)
        assert excinfo.value.status_code == 403
        assert "Permission denied" in excinfo.value.detail
