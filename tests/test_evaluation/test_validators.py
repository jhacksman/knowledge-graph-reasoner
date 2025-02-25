"""Tests for evaluation validators module."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.node import Node
from src.models.edge import Edge
from src.evaluation.validators import (
    EntityValidator,
    RelationshipValidator,
    DomainValidator,
    CoherenceValidator,
    CrossDomainValidator,
    GraphValidator
)


@pytest.fixture
def mock_graph_manager():
    """Create a mock graph manager."""
    manager = AsyncMock()
    
    # Mock get_graph_state
    manager.get_graph_state.return_value = {
        "timestamp": 1234567890,
        "node_count": 10,
        "edge_count": 15,
        "modularity": 0.75,
        "avg_path_length": 2.5,
        "diameter": 5,
        "communities": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9", "10"]],
        "bridge_nodes": ["3", "6"],
        "centrality": {
            "1": 0.1, "2": 0.2, "3": 0.8, "4": 0.3, 
            "5": 0.2, "6": 0.7, "7": 0.4, "8": 0.3,
            "9": 0.2, "10": 0.1
        }
    }
    
    # Mock get_concept
    async def mock_get_concept(node_id):
        return Node(
            id=node_id,
            content=f"Content for node {node_id}",
            metadata={"embedding": np.random.rand(10).tolist()}
        )
    
    manager.get_concept = mock_get_concept
    
    # Mock get_all_concepts
    async def mock_get_all_concepts():
        for i in range(1, 11):
            yield Node(
                id=str(i),
                content=f"Content for node {i}",
                metadata={"embedding": np.random.rand(10).tolist()}
            )
    
    manager.get_all_concepts = mock_get_all_concepts
    
    # Mock get_all_relationships
    async def mock_get_all_relationships():
        edges = [
            ("1", "2", "related_to"),
            ("2", "3", "causes"),
            ("3", "4", "part_of"),
            ("4", "5", "related_to"),
            ("5", "6", "causes")
        ]
        
        for source, target, rel_type in edges:
            yield Edge(
                source=source,
                target=target,
                type=rel_type,
                metadata={"confidence": 0.9}
            )
    
    manager.get_all_relationships = mock_get_all_relationships
    
    return manager


@pytest.mark.asyncio
async def test_entity_validator():
    """Test entity validator."""
    validator = EntityValidator(
        min_content_length=5,
        max_content_length=100,
        required_metadata=["embedding"]
    )
    
    # Test valid node
    valid_node = Node(
        id="1",
        content="This is a valid node with sufficient content",
        metadata={"embedding": [0.1, 0.2, 0.3]}
    )
    
    result = await validator.validate_entity(valid_node)
    assert result["valid"] is True
    assert len(result["issues"]) == 0
    
    # Test invalid node - content too short
    short_node = Node(
        id="2",
        content="Short",
        metadata={"embedding": [0.1, 0.2, 0.3]}
    )
    
    result = await validator.validate_entity(short_node)
    assert result["valid"] is False
    assert len(result["issues"]) > 0
    assert any("content_length" in issue["type"] for issue in result["issues"])
    
    # Test invalid node - missing metadata
    missing_metadata_node = Node(
        id="3",
        content="This node is missing required metadata",
        metadata={}
    )
    
    result = await validator.validate_entity(missing_metadata_node)
    assert result["valid"] is False
    assert len(result["issues"]) > 0
    assert any("missing_metadata" in issue["type"] for issue in result["issues"])


@pytest.mark.asyncio
async def test_relationship_validator():
    """Test relationship validator."""
    validator = RelationshipValidator(
        valid_relationship_types=["related_to", "causes", "part_of"],
        min_confidence=0.7
    )
    
    # Test valid edge
    valid_edge = Edge(
        source="1",
        target="2",
        type="related_to",
        metadata={"confidence": 0.8}
    )
    
    result = await validator.validate_relationship(valid_edge)
    assert result["valid"] is True
    assert len(result["issues"]) == 0
    
    # Test invalid edge - invalid relationship type
    invalid_type_edge = Edge(
        source="1",
        target="2",
        type="unknown_relation",
        metadata={"confidence": 0.8}
    )
    
    result = await validator.validate_relationship(invalid_type_edge)
    assert result["valid"] is False
    assert len(result["issues"]) > 0
    assert any("invalid_relationship_type" in issue["type"] for issue in result["issues"])
    
    # Test invalid edge - low confidence
    low_confidence_edge = Edge(
        source="1",
        target="2",
        type="related_to",
        metadata={"confidence": 0.5}
    )
    
    result = await validator.validate_relationship(low_confidence_edge)
    assert result["valid"] is False
    assert len(result["issues"]) > 0
    assert any("low_confidence" in issue["type"] for issue in result["issues"])


@pytest.mark.asyncio
async def test_domain_validator():
    """Test domain validator."""
    validator = DomainValidator(
        domain="science",
        domain_keywords=["physics", "quantum", "biology", "chemistry"]
    )
    
    # Test relevant node
    relevant_node = Node(
        id="1",
        content="Quantum mechanics is a fundamental theory in physics",
        metadata={}
    )
    
    result = await validator.validate_domain_relevance(relevant_node)
    assert result["relevant"] is True
    assert len(result["matches"]) > 0
    assert "quantum" in result["matches"]
    
    # Test irrelevant node
    irrelevant_node = Node(
        id="2",
        content="This node has nothing to do with the domain",
        metadata={}
    )
    
    result = await validator.validate_domain_relevance(irrelevant_node)
    assert result["relevant"] is False
    assert len(result["matches"]) == 0


@pytest.mark.asyncio
async def test_coherence_validator():
    """Test coherence validator."""
    validator = CoherenceValidator(min_coherence_score=0.7)
    
    # Test coherent community
    # Create nodes with similar embeddings
    coherent_nodes = []
    base_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    for i in range(5):
        # Add small random noise to create similar embeddings
        noise = np.random.normal(0, 0.05, 5)
        embedding = base_embedding + noise
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        coherent_nodes.append(Node(
            id=str(i),
            content=f"Node {i}",
            metadata={"embedding": embedding.tolist()}
        ))
    
    result = await validator.validate_community_coherence(coherent_nodes)
    assert result["coherent"] is True
    assert result["coherence_score"] >= 0.7
    
    # Test incoherent community
    # Create nodes with random embeddings
    incoherent_nodes = []
    
    for i in range(5):
        embedding = np.random.rand(5)
        embedding = embedding / np.linalg.norm(embedding)
        
        incoherent_nodes.append(Node(
            id=str(i),
            content=f"Node {i}",
            metadata={"embedding": embedding.tolist()}
        ))
    
    # Force low coherence for testing
    with patch('numpy.dot', return_value=0.3):
        result = await validator.validate_community_coherence(incoherent_nodes)
        assert result["coherent"] is False
        assert result["coherence_score"] < 0.7


@pytest.mark.asyncio
async def test_cross_domain_validator():
    """Test cross-domain validator."""
    validator = CrossDomainValidator(
        domains={
            "science": ["physics", "quantum", "biology", "chemistry"],
            "technology": ["computer", "software", "hardware", "algorithm"],
            "humanities": ["philosophy", "history", "literature", "art"]
        },
        min_cross_domain_connections=2
    )
    
    # Create nodes from different domains
    nodes = [
        Node(id="1", content="Quantum mechanics is a fundamental theory in physics", metadata={}),
        Node(id="2", content="Computer algorithms are used in quantum computing", metadata={}),
        Node(id="3", content="The philosophy of science examines the foundations of scientific knowledge", metadata={}),
        Node(id="4", content="Software engineering principles apply to quantum computing", metadata={}),
        Node(id="5", content="The history of physics includes many revolutionary discoveries", metadata={})
    ]
    
    # Create edges connecting different domains
    edges = [
        Edge(source="1", target="2", type="related_to", metadata={}),  # science -> technology
        Edge(source="2", target="3", type="related_to", metadata={}),  # technology -> humanities
        Edge(source="3", target="4", type="related_to", metadata={}),  # humanities -> technology
        Edge(source="4", target="5", type="related_to", metadata={}),  # technology -> humanities
        Edge(source="5", target="1", type="related_to", metadata={})   # humanities -> science
    ]
    
    result = await validator.validate_cross_domain_connections(nodes, edges)
    assert result["valid"] is True
    assert result["cross_domain_connections"] >= 2
    
    # Test with insufficient cross-domain connections
    validator = CrossDomainValidator(
        domains={
            "science": ["physics", "quantum", "biology", "chemistry"],
            "technology": ["computer", "software", "hardware", "algorithm"]
        },
        min_cross_domain_connections=10  # Set high threshold
    )
    
    result = await validator.validate_cross_domain_connections(nodes, edges)
    assert result["valid"] is False
    assert result["cross_domain_connections"] < 10


@pytest.mark.asyncio
async def test_graph_validator(mock_graph_manager):
    """Test comprehensive graph validator."""
    # Create validators
    entity_validator = EntityValidator(min_content_length=5)
    relationship_validator = RelationshipValidator(min_confidence=0.7)
    coherence_validator = CoherenceValidator(min_coherence_score=0.6)
    domain_validators = [
        DomainValidator(domain="science", domain_keywords=["physics", "quantum"]),
        DomainValidator(domain="technology", domain_keywords=["computer", "software"])
    ]
    cross_domain_validator = CrossDomainValidator(
        domains={
            "science": ["physics", "quantum"],
            "technology": ["computer", "software"]
        },
        min_cross_domain_connections=1
    )
    
    # Create graph validator
    validator = GraphValidator(
        entity_validator=entity_validator,
        relationship_validator=relationship_validator,
        coherence_validator=coherence_validator,
        domain_validators=domain_validators,
        cross_domain_validator=cross_domain_validator
    )
    
    # Test validate_graph
    result = await validator.validate_graph(mock_graph_manager)
    
    # Check result structure
    assert "entity_validation" in result
    assert "relationship_validation" in result
    assert "coherence_validation" in result
    assert "domain_validation" in result
    assert "cross_domain_validation" in result
    
    # Check entity validation
    assert "total" in result["entity_validation"]
    assert "valid" in result["entity_validation"]
    assert "invalid" in result["entity_validation"]
    
    # Check relationship validation
    assert "total" in result["relationship_validation"]
    assert "valid" in result["relationship_validation"]
    assert "invalid" in result["relationship_validation"]
