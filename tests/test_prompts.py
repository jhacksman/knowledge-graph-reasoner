"""Tests for structured prompts."""
import pytest
import json

from src.reasoning.prompts import (
    CONCEPT_GENERATION_PROMPT,
    RELATIONSHIP_INFERENCE_PROMPT,
    BRIDGE_NODE_PROMPT,
    STABILITY_CHECK_PROMPT
)


def test_concept_generation_prompt():
    """Test concept generation prompt formatting."""
    state = {
        "modularity": 0.75,
        "avg_path_length": 4.8,
        "bridge_nodes": ["node1", "node2"]
    }
    
    prompt = CONCEPT_GENERATION_PROMPT.format(
        modularity=state["modularity"],
        avg_path_length=state["avg_path_length"],
        bridge_nodes=len(state["bridge_nodes"]),
        seed_concept="test concept"
    )
    
    assert "modularity: 0.75" in prompt.lower()
    assert "average path length: 4.8" in prompt.lower()
    assert "test concept" in prompt
    assert "format: return a list of concepts" in prompt.lower()


def test_relationship_inference_prompt():
    """Test relationship inference prompt formatting."""
    prompt = RELATIONSHIP_INFERENCE_PROMPT.format(
        concept_a="concept A",
        concept_b="concept B",
        graph_context="test context"
    )
    
    assert "Concept A:" in prompt
    assert "Concept B:" in prompt
    assert "test context" in prompt
    assert "Format: Return as JSON" in prompt
    assert "exists (bool)" in prompt
    assert "type (str)" in prompt
    assert "confidence (float)" in prompt


def test_bridge_node_prompt():
    """Test bridge node prompt formatting."""
    prompt = BRIDGE_NODE_PROMPT.format(
        domain_a="Physics",
        domain_b="Chemistry",
        bridge_nodes=["node1", "node2"],
        modularity=0.75
    )
    
    assert "Domain A: Physics" in prompt
    assert "Domain B: Chemistry" in prompt
    assert "Current bridge nodes: ['node1', 'node2']" in prompt
    assert "Modularity score: 0.75" in prompt
    assert "Format: Return a list of bridge concepts" in prompt


def test_stability_check_prompt():
    """Test stability check prompt formatting."""
    state = {
        "modularity": 0.75,
        "avg_path_length": 4.8,
        "diameter": 17.0,
        "bridge_node_count": 5
    }
    
    prompt = STABILITY_CHECK_PROMPT.format(
        modularity=state["modularity"],
        avg_path_length=state["avg_path_length"],
        diameter=state["diameter"],
        bridge_node_count=state["bridge_node_count"]
    )
    
    assert "Current modularity: 0.75" in prompt
    assert "Average path length: 4.8" in prompt
    assert "Graph diameter: 17.0" in prompt
    assert "Bridge node count: 5" in prompt
    assert "Path length: 4.5-5.0" in prompt
    assert "Diameter: 16-18" in prompt
    assert "Format: Return as JSON" in prompt
    
    # Verify JSON schema fields
    schema_fields = [
        '"stable": bool',
        '"metrics_in_range": List[str]',
        '"improvements_needed": List[str]',
        '"recommendations": List[str]'
    ]
    for field in schema_fields:
        assert field in prompt
