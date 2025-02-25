"""Tests for vector store base functionality."""
import pytest
from typing import List

from src.models.node import Node
from src.models.edge import Edge
from src.vector_store.base import deduplicate_nodes, deduplicate_edges


def test_deduplicate_nodes():
    """Test node deduplication."""
    nodes = [
        Node(id="1", content="test1"),
        Node(id="2", content="test2"),
        Node(id="3", content="test1"),  # Duplicate content
        Node(id="1", content="test1"),  # Duplicate id and content
    ]
    
    unique_nodes = deduplicate_nodes(nodes)
    
    # Should only keep first occurrence of each unique content+id
    assert len(unique_nodes) == 2
    assert unique_nodes[0].id == "1"
    assert unique_nodes[1].id == "2"


def test_deduplicate_edges():
    """Test edge deduplication."""
    edges = [
        Edge(source="1", target="2", type="related"),
        Edge(source="2", target="3", type="similar"),
        Edge(source="1", target="2", type="related"),  # Duplicate
        Edge(source="2", target="3", type="similar"),  # Duplicate
    ]
    
    unique_edges = deduplicate_edges(edges)
    
    # Should only keep first occurrence of each unique source+target+type
    assert len(unique_edges) == 2
    assert unique_edges[0].source == "1"
    assert unique_edges[0].target == "2"
    assert unique_edges[1].source == "2"
    assert unique_edges[1].target == "3"
