"""Validation module for knowledge graph reasoner."""
from typing import List, Dict, Any, Optional, Set, Tuple, AsyncIterator
import logging
import numpy as np

from ..models.node import Node
from ..models.edge import Edge
from ..graph.manager import GraphManager
from ..reasoning.llm import VeniceLLM

log = logging.getLogger(__name__)


class EntityValidator:
    """Validates entities in knowledge graph."""
    
    def __init__(
        self,
        min_content_length: int = 10,
        max_content_length: int = 10000,
        required_metadata: Optional[List[str]] = None,
        llm: Optional[VeniceLLM] = None
    ):
        """Initialize validator."""
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.required_metadata = required_metadata or []
        self.llm = llm
    
    async def validate_entity(self, node: Node) -> Dict[str, Any]:
        """Validate entity."""
        results: Dict[str, Any] = {
            "id": node.id,
            "valid": True,
            "issues": []
        }
        
        # Check content length
        if len(node.content) < self.min_content_length:
            results["valid"] = False
            results["issues"].append({
                "type": "content_length",
                "message": f"Content too short (min {self.min_content_length} chars)"
            })
        
        if len(node.content) > self.max_content_length:
            results["valid"] = False
            results["issues"].append({
                "type": "content_length",
                "message": f"Content too long (max {self.max_content_length} chars)"
            })
        
        # Check required metadata
        for field in self.required_metadata:
            if field not in node.metadata:
                results["valid"] = False
                results["issues"].append({
                    "type": "missing_metadata",
                    "message": f"Missing required metadata field: {field}"
                })
        
        return results


class RelationshipValidator:
    """Validates relationships in knowledge graph."""
    
    def __init__(
        self,
        valid_relationship_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        llm: Optional[VeniceLLM] = None
    ):
        """Initialize validator."""
        self.valid_relationship_types = valid_relationship_types
        self.min_confidence = min_confidence
        self.llm = llm
    
    async def validate_relationship(
        self, 
        edge: Edge, 
        source_node: Optional[Node] = None,
        target_node: Optional[Node] = None
    ) -> Dict[str, Any]:
        """Validate relationship."""
        results: Dict[str, Any] = {
            "source": edge.source,
            "target": edge.target,
            "type": edge.type,
            "valid": True,
            "issues": []
        }
        
        # Check relationship type
        if self.valid_relationship_types and edge.type not in self.valid_relationship_types:
            results["valid"] = False
            results["issues"].append({
                "type": "invalid_relationship_type",
                "message": f"Invalid relationship type: {edge.type}"
            })
        
        # Check confidence if available
        if "confidence" in edge.metadata:
            confidence = edge.metadata["confidence"]
            if confidence < self.min_confidence:
                results["valid"] = False
                results["issues"].append({
                    "type": "low_confidence",
                    "message": f"Confidence too low: {confidence} (min {self.min_confidence})"
                })
        
        return results


class DomainValidator:
    """Validates domain-specific knowledge."""
    
    def __init__(
        self,
        domain: str,
        domain_keywords: List[str],
        llm: Optional[VeniceLLM] = None
    ):
        """Initialize validator."""
        self.domain = domain
        self.domain_keywords = domain_keywords
        self.llm = llm
    
    async def validate_domain_relevance(self, node: Node) -> Dict[str, Any]:
        """Validate domain relevance."""
        results: Dict[str, Any] = {
            "id": node.id,
            "domain": self.domain,
            "relevant": False,
            "confidence": 0.0,
            "matches": []
        }
        
        # Check keywords
        keyword_matches = []
        for keyword in self.domain_keywords:
            if keyword.lower() in node.content.lower():
                keyword_matches.append(keyword)
        
        # Determine relevance
        if keyword_matches:
            results["relevant"] = True
            results["matches"] = keyword_matches
            results["confidence"] = min(1.0, len(keyword_matches) / 10)  # Simple heuristic
        
        return results


class CoherenceValidator:
    """Validates knowledge coherence."""
    
    def __init__(
        self,
        min_coherence_score: float = 0.7,
        llm: Optional[VeniceLLM] = None
    ):
        """Initialize validator."""
        self.min_coherence_score = min_coherence_score
        self.llm = llm
    
    async def validate_community_coherence(
        self,
        nodes: List[Node]
    ) -> Dict[str, Any]:
        """Validate community coherence."""
        results: Dict[str, Any] = {
            "node_count": len(nodes),
            "coherent": True,
            "coherence_score": 0.0,
            "issues": []
        }
        
        if len(nodes) < 2:
            # Single node is perfectly coherent
            results["coherence_score"] = 1.0
            return results
        
        try:
            # Get embeddings
            embeddings = []
            for node in nodes:
                if "embedding" in node.metadata:
                    embeddings.append(node.metadata["embedding"])
            
            if len(embeddings) < 2:
                results["coherent"] = False
                results["issues"].append({
                    "type": "missing_embeddings",
                    "message": "Not enough nodes have embeddings"
                })
                return results
            
            # Compute pairwise cosine similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            
            # Average similarity is the coherence score
            coherence_score = float(np.mean(similarities))
            results["coherence_score"] = coherence_score
            
            # Check if coherent
            if coherence_score < self.min_coherence_score:
                results["coherent"] = False
                results["issues"].append({
                    "type": "low_coherence",
                    "message": f"Coherence score too low: {coherence_score} (min {self.min_coherence_score})"
                })
            
            return results
        except Exception as e:
            log.error(f"Failed to validate community coherence: {e}")
            results["coherent"] = False
            results["issues"].append({
                "type": "error",
                "message": f"Error validating coherence: {str(e)}"
            })
            return results


class CrossDomainValidator:
    """Validates cross-domain connections."""
    
    def __init__(
        self,
        domains: Dict[str, List[str]],
        min_cross_domain_connections: int = 1,
        llm: Optional[VeniceLLM] = None
    ):
        """Initialize validator.
        
        Args:
            domains: Dictionary mapping domain names to lists of keywords
            min_cross_domain_connections: Minimum number of cross-domain connections
            llm: LLM client for validation (optional)
        """
        self.domains = domains
        self.min_cross_domain_connections = min_cross_domain_connections
        self.llm = llm
    
    async def validate_cross_domain_connections(
        self,
        nodes: List[Node],
        edges: List[Edge]
    ) -> Dict[str, Any]:
        """Validate cross-domain connections.
        
        Args:
            nodes: List of nodes
            edges: List of edges
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results: Dict[str, Any] = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "valid": True,
            "cross_domain_connections": 0,
            "domain_connections": {},
            "issues": []
        }
        
        try:
            # Classify nodes by domain
            node_domains: Dict[str, str] = {}
            
            for node in nodes:
                # Check each domain
                node_domain = None
                max_matches = 0
                
                for domain, keywords in self.domains.items():
                    matches = 0
                    for keyword in keywords:
                        if keyword.lower() in node.content.lower():
                            matches += 1
                    
                    if matches > max_matches:
                        max_matches = matches
                        node_domain = domain
                
                if node_domain:
                    node_domains[node.id] = node_domain
            
            # Count cross-domain connections
            domain_connections: Dict[str, Dict[str, int]] = {}
            cross_domain_count = 0
            
            for edge in edges:
                source_domain = node_domains.get(edge.source)
                target_domain = node_domains.get(edge.target)
                
                if source_domain and target_domain and source_domain != target_domain:
                    cross_domain_count += 1
                    
                    # Track domain connections
                    if source_domain not in domain_connections:
                        domain_connections[source_domain] = {}
                    
                    if target_domain not in domain_connections[source_domain]:
                        domain_connections[source_domain][target_domain] = 0
                    
                    domain_connections[source_domain][target_domain] += 1
                    
                    # Also track in reverse direction for undirected analysis
                    if target_domain not in domain_connections:
                        domain_connections[target_domain] = {}
                    
                    if source_domain not in domain_connections[target_domain]:
                        domain_connections[target_domain][source_domain] = 0
                    
                    domain_connections[target_domain][source_domain] += 1
            
            # Update results
            results["cross_domain_connections"] = cross_domain_count
            results["domain_connections"] = domain_connections
            
            # Check if valid
            if cross_domain_count < self.min_cross_domain_connections:
                results["valid"] = False
                results["issues"].append({
                    "type": "insufficient_cross_domain",
                    "message": f"Insufficient cross-domain connections: {cross_domain_count} (min {self.min_cross_domain_connections})"
                })
            
            return results
        except Exception as e:
            log.error(f"Failed to validate cross-domain connections: {e}")
            results["valid"] = False
            results["issues"].append({
                "type": "error",
                "message": f"Error validating cross-domain connections: {str(e)}"
            })
            return results


class GraphValidator:
    """Comprehensive graph validator."""
    
    def __init__(
        self,
        entity_validator: Optional[EntityValidator] = None,
        relationship_validator: Optional[RelationshipValidator] = None,
        coherence_validator: Optional[CoherenceValidator] = None,
        domain_validators: Optional[List[DomainValidator]] = None,
        cross_domain_validator: Optional[CrossDomainValidator] = None
    ):
        """Initialize validator.
        
        Args:
            entity_validator: Entity validator
            relationship_validator: Relationship validator
            coherence_validator: Coherence validator
            domain_validators: List of domain validators
            cross_domain_validator: Cross-domain validator
        """
        self.entity_validator = entity_validator or EntityValidator()
        self.relationship_validator = relationship_validator or RelationshipValidator()
        self.coherence_validator = coherence_validator or CoherenceValidator()
        self.domain_validators = domain_validators or []
        self.cross_domain_validator = cross_domain_validator
    
    async def validate_graph(
        self,
        graph_manager: GraphManager
    ) -> Dict[str, Any]:
        """Validate entire graph.
        
        Args:
            graph_manager: Graph manager
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results: Dict[str, Any] = {
            "timestamp": None,
            "valid": True,
            "entity_validation": {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "issues": []
            },
            "relationship_validation": {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "issues": []
            },
            "coherence_validation": {},
            "domain_validation": {},
            "cross_domain_validation": {}
        }
        
        try:
            # Get all nodes and edges
            nodes = []
            for node in await graph_manager.get_concepts():
                nodes.append(node)
            
            edges = []
            for edge in await graph_manager.get_relationships():
                edges.append(edge)
            
            # Get timestamp
            state = await graph_manager.get_graph_state()
            results["timestamp"] = state.get("timestamp", 0)
            
            # Validate entities
            for node in nodes:
                entity_result = await self.entity_validator.validate_entity(node)
                results["entity_validation"]["total"] += 1
                
                if entity_result["valid"]:
                    results["entity_validation"]["valid"] += 1
                else:
                    results["entity_validation"]["invalid"] += 1
                    results["valid"] = False
                    
                    for issue in entity_result["issues"]:
                        results["entity_validation"]["issues"].append({
                            "node_id": node.id,
                            "issue": issue
                        })
            
            # Validate relationships
            for edge in edges:
                # Get source and target nodes
                source_node = None
                target_node = None
                
                for node in nodes:
                    if node.id == edge.source:
                        source_node = node
                    elif node.id == edge.target:
                        target_node = node
                
                relationship_result = await self.relationship_validator.validate_relationship(
                    edge, source_node, target_node
                )
                
                results["relationship_validation"]["total"] += 1
                
                if relationship_result["valid"]:
                    results["relationship_validation"]["valid"] += 1
                else:
                    results["relationship_validation"]["invalid"] += 1
                    results["valid"] = False
                    
                    for issue in relationship_result["issues"]:
                        results["relationship_validation"]["issues"].append({
                            "source": edge.source,
                            "target": edge.target,
                            "type": edge.type,
                            "issue": issue
                        })
            
            # Validate coherence
            # Get communities
            communities = state.get("communities", [])
            
            if communities and self.coherence_validator:
                community_results = []
                
                for i, community_ids in enumerate(communities):
                    # Get nodes in community
                    community_nodes = []
                    for node_id in community_ids:
                        for node in nodes:
                            if node.id == node_id:
                                community_nodes.append(node)
                                break
                    
                    # Validate community
                    coherence_result = await self.coherence_validator.validate_community_coherence(
                        community_nodes
                    )
                    
                    coherence_result["community_id"] = i
                    community_results.append(coherence_result)
                    
                    if not coherence_result["coherent"]:
                        results["valid"] = False
                
                # Add to results
                results["coherence_validation"] = {
                    "communities": community_results,
                    "avg_coherence": np.mean([r["coherence_score"] for r in community_results]) if community_results else 0.0
                }
            
            # Validate domains
            if self.domain_validators:
                domain_results: Dict[str, Any] = {}
                
                for validator in self.domain_validators:
                    domain_nodes = []
                    
                    for node in nodes:
                        domain_result = await validator.validate_domain_relevance(node)
                        if domain_result["relevant"]:
                            domain_nodes.append(node)
                    
                    domain_results[validator.domain] = {
                        "total_nodes": len(nodes),
                        "relevant_nodes": len(domain_nodes),
                        "coverage": len(domain_nodes) / len(nodes) if nodes else 0.0
                    }
                
                results["domain_validation"] = domain_results
            
            # Validate cross-domain connections
            if self.cross_domain_validator:
                cross_domain_result = await self.cross_domain_validator.validate_cross_domain_connections(
                    nodes, edges
                )
                
                results["cross_domain_validation"] = cross_domain_result
                
                if not cross_domain_result["valid"]:
                    results["valid"] = False
            
            return results
        except Exception as e:
            log.error(f"Failed to validate graph: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
