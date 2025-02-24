"""Tests for core data models."""
import pytest
from src.models.node import Node
from src.models.edge import Edge


def test_node_creation():
    """Test basic node creation and validation."""
    # Test basic creation
    node = Node(
        id="test_node",
        content="Test content"
    )
    assert node.id == "test_node"
    assert node.content == "Test content"
    assert node.metadata == {}
    
    # Test with metadata
    node = Node(
        id="test_node",
        content="Test content",
        metadata={"key": "value"}
    )
    assert node.metadata["key"] == "value"
    
    # Test validation
    with pytest.raises(ValueError):
        Node(content="Missing ID")  # Missing required field
    
    with pytest.raises(ValueError):
        Node(id="test_node")  # Missing required field


def test_edge_creation():
    """Test basic edge creation and validation."""
    # Test basic creation
    edge = Edge(
        source="node1",
        target="node2",
        type="related"
    )
    assert edge.source == "node1"
    assert edge.target == "node2"
    assert edge.type == "related"
    assert edge.metadata == {}
    
    # Test with metadata
    edge = Edge(
        source="node1",
        target="node2",
        type="related",
        metadata={"weight": 0.8}
    )
    assert edge.metadata["weight"] == 0.8
    
    # Test validation
    with pytest.raises(ValueError):
        Edge(target="node2", type="related")  # Missing required field
    
    with pytest.raises(ValueError):
        Edge(source="node1", type="related")  # Missing required field
    
    with pytest.raises(ValueError):
        Edge(source="node1", target="node2")  # Missing required field
