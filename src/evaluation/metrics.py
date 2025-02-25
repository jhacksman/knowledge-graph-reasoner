"""Core evaluation metrics for knowledge graphs."""
from typing import List, Dict, Any, Optional, Set, Tuple, AsyncIterator, Collection
import numpy as np
import networkx as nx
import logging
from collections import defaultdict

from ..models.node import Node
from ..models.edge import Edge
from ..graph.manager import GraphManager
from ..reasoning.llm import VeniceLLM

log = logging.getLogger(__name__)


class GraphEvaluator:
    """Evaluates knowledge graph quality and reasoning performance."""
    
    def __init__(
        self,
        graph_manager: GraphManager,
        llm: Optional[VeniceLLM] = None,
        evaluation_interval: int = 5,
        domains: Optional[List[str]] = None
    ):
        """Initialize evaluator.
        
        Args:
            graph_manager: Graph manager instance
            llm: LLM client for validation (optional)
            evaluation_interval: Number of iterations between evaluations
            domains: List of knowledge domains to track
        """
        self.graph_manager = graph_manager
        self.llm = llm
        self.evaluation_interval = evaluation_interval
        self.domains = domains or []
        self.history: List[Dict[str, Any]] = []
        
    async def compute_semantic_coherence(self, community: List[Node]) -> float:
        """Compute semantic coherence score for a community.
        
        Args:
            community: List of nodes in the community
            
        Returns:
            float: Coherence score (0-1)
        """
        try:
            if len(community) < 2:
                return 1.0  # Single node is perfectly coherent
            
            # Get embeddings
            embeddings = []
            for node in community:
                if "embedding" in node.metadata:
                    embeddings.append(node.metadata["embedding"])
            
            if len(embeddings) < 2:
                return 0.0  # Not enough embeddings
            
            # Compute pairwise cosine similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            
            # Average similarity is the coherence score
            return float(np.mean(similarities))
        except Exception as e:
            log.error(f"Failed to compute semantic coherence: {e}")
            return 0.0
    
    async def validate_relationship(self, edge: Edge, source_node: Node, target_node: Node) -> float:
        """Validate relationship using LLM.
        
        Args:
            edge: Edge to validate
            source_node: Source node
            target_node: Target node
            
        Returns:
            float: Validity score (0-1)
        """
        if not self.llm:
            return 1.0  # No LLM available, assume valid
        
        try:
            # Construct prompt
            prompt = f"""Evaluate if this relationship is valid:
            
            Source: {source_node.content}
            Target: {target_node.content}
            Relationship: {edge.type}
            
            Is this relationship valid? Respond with a score from 0 to 1, where:
            - 0 means completely invalid
            - 0.5 means uncertain
            - 1 means definitely valid
            
            Score:"""
            
            # Get LLM response
            response = await self.llm.generate([{
                "role": "system",
                "content": "You are a knowledge validation system."
            }, {
                "role": "user",
                "content": prompt
            }])
            
            # Parse response
            content = response["choices"][0]["message"]["content"]
            
            # Extract score
            try:
                # Look for a number in the response
                import re
                match = re.search(r'(\d+(\.\d+)?)', content)
                if match:
                    score = float(match.group(1))
                    # Ensure score is in [0, 1]
                    return max(0.0, min(1.0, score))
                else:
                    # Default to uncertain if no number found
                    return 0.5
            except Exception:
                return 0.5  # Default to uncertain
        except Exception as e:
            log.error(f"Failed to validate relationship: {e}")
            return 0.5  # Default to uncertain
    
    async def compute_domain_coverage(self) -> Dict[str, float]:
        """Compute coverage across configured knowledge domains.
        
        Returns:
            Dict[str, float]: Domain to coverage score mapping
        """
        if not self.domains:
            return {}
        
        try:
            # Get all nodes
            nodes = []
            for node in await self.graph_manager.get_concept():
                nodes.append(node)
            
            if not nodes:
                return {domain: 0.0 for domain in self.domains}
            
            # Compute domain coverage
            coverage = {}
            for domain in self.domains:
                # Count nodes related to domain
                domain_nodes = 0
                for node in nodes:
                    # Check if domain is mentioned in content or metadata
                    if (
                        domain.lower() in node.content.lower() or
                        any(domain.lower() in str(v).lower() for v in node.metadata.values())
                    ):
                        domain_nodes += 1
                
                # Coverage is percentage of nodes related to domain
                coverage[domain] = domain_nodes / len(nodes)
            
            return coverage
        except Exception as e:
            log.error(f"Failed to compute domain coverage: {e}")
            return {domain: 0.0 for domain in self.domains}
    
    async def compute_interdisciplinary_metrics(self) -> Dict[str, float]:
        """Compute interdisciplinary connection metrics.
        
        Returns:
            Dict[str, float]: Metric name to value mapping
        """
        try:
            # Get graph state
            state = await self.graph_manager.get_graph_state()
            
            # Get communities
            communities = state.get("communities", [])
            if not communities:
                return {
                    "cross_community_edges": 0.0,
                    "interdisciplinary_ratio": 0.0,
                    "bridge_node_centrality": 0.0
                }
            
            # Compute cross-community edges
            cross_edges = 0
            total_edges = 0
            
            # Get all edges
            edges = []
            for edge in await self.graph_manager.get_relationship():
                edges.append(edge)
                total_edges += 1
            
            # Map nodes to communities
            node_to_community = {}
            for i, community in enumerate(communities):
                for node_id in community:
                    node_to_community[node_id] = i
            
            # Count cross-community edges
            for edge in edges:
                if (
                    edge.source in node_to_community and
                    edge.target in node_to_community and
                    node_to_community[edge.source] != node_to_community[edge.target]
                ):
                    cross_edges += 1
            
            # Compute interdisciplinary ratio
            interdisciplinary_ratio = cross_edges / max(1, total_edges)
            
            # Compute bridge node centrality
            bridge_nodes = state.get("bridge_nodes", [])
            bridge_centrality = 0.0
            if bridge_nodes:
                # Get centrality scores
                centrality = state.get("centrality", {})
                if centrality:
                    # Average centrality of bridge nodes
                    bridge_centrality = sum(
                        centrality.get(node, 0.0) for node in bridge_nodes
                    ) / len(bridge_nodes)
            
            return {
                "cross_community_edges": cross_edges,
                "interdisciplinary_ratio": interdisciplinary_ratio,
                "bridge_node_centrality": bridge_centrality
            }
        except Exception as e:
            log.error(f"Failed to compute interdisciplinary metrics: {e}")
            return {
                "cross_community_edges": 0.0,
                "interdisciplinary_ratio": 0.0,
                "bridge_node_centrality": 0.0
            }
    
    async def compute_novelty_score(self, node: Node) -> float:
        """Compute novelty/"surprise" score for a concept.
        
        Args:
            node: Node to evaluate
            
        Returns:
            float: Novelty score (0-1)
        """
        try:
            # Get similar concepts
            if "embedding" not in node.metadata:
                return 0.0
            
            embedding = node.metadata["embedding"]
            similar = await self.graph_manager.get_similar_concepts(
                embedding,
                k=5,
                threshold=0.0  # No threshold to get 5 most similar
            )
            
            if not similar:
                return 1.0  # No similar concepts, completely novel
            
            # Compute average similarity
            similarities = []
            for similar_node in similar:
                if "embedding" in similar_node.metadata:
                    sim = np.dot(embedding, similar_node.metadata["embedding"]) / (
                        np.linalg.norm(embedding) * np.linalg.norm(similar_node.metadata["embedding"])
                    )
                    similarities.append(sim)
            
            if not similarities:
                return 1.0
            
            # Novelty is inverse of average similarity
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity
        except Exception as e:
            log.error(f"Failed to compute novelty score: {e}")
            return 0.0
    
    async def evaluate(self, iteration: int) -> Dict[str, Any]:
        """Perform comprehensive evaluation.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Skip if not at evaluation interval
        if iteration % self.evaluation_interval != 0:
            return {}
        
        try:
            # Get graph state
            state = await self.graph_manager.get_graph_state()
            
            # Basic metrics from graph state
            results = {
                "iteration": iteration,
                "timestamp": state.get("timestamp", 0),
                "node_count": state.get("node_count", 0),
                "edge_count": state.get("edge_count", 0),
                "modularity": state.get("modularity", 0.0),
                "avg_path_length": state.get("avg_path_length", 0.0),
                "diameter": state.get("diameter", 0.0)
            }
            
            # Get communities
            communities = state.get("communities", [])
            
            # Compute semantic coherence for each community
            community_coherence = []
            for community_ids in communities:
                # Get nodes in community
                community_nodes = []
                for node_id in community_ids:
                    node = await self.graph_manager.get_concept(node_id)
                    if node:
                        community_nodes.append(node)
                
                # Compute coherence
                coherence = await self.compute_semantic_coherence(community_nodes)
                community_coherence.append(coherence)
            
            # Add community coherence metrics
            if community_coherence:
                results["avg_community_coherence"] = np.mean(community_coherence)
                results["min_community_coherence"] = min(community_coherence)
                results["max_community_coherence"] = max(community_coherence)
            
            # Compute domain coverage
            domain_coverage = await self.compute_domain_coverage()
            if domain_coverage:
                results["domain_coverage"] = domain_coverage
            
            # Compute interdisciplinary metrics
            interdisciplinary = await self.compute_interdisciplinary_metrics()
            results.update(interdisciplinary)
            
            # Compute novelty scores for recent nodes
            # In a real implementation, this would track recently added nodes
            # For now, just compute for a sample of nodes
            novelty_scores = []
            node_count = 0
            for node in await self.graph_manager.get_concept():
                novelty = await self.compute_novelty_score(node if node is not None else Node(id='dummy', content='', metadata={}))  # type: ignore  # type: ignore
                novelty_scores.append(novelty)
                node_count += 1
                if node_count >= 10:  # Limit to 10 nodes for performance
                    break
            
            if novelty_scores:
                results["avg_novelty"] = np.mean(novelty_scores)
            
            # Add to history
            self.history.append(results)
            
            return results
        except Exception as e:
            log.error(f"Failed to evaluate graph: {e}")
            return {
                "iteration": iteration,
                "error": str(e)
            }
    
    async def get_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history.
        
        Returns:
            List[Dict[str, Any]]: Evaluation history
        """
        return self.history
    
    async def compare_iterations(self, iter1: int, iter2: int) -> Dict[str, Any]:
        """Compare metrics between two iterations.
        
        Args:
            iter1: First iteration
            iter2: Second iteration
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            # Find iterations in history
            iter1_data = None
            iter2_data = None
            
            for entry in self.history:
                if entry.get("iteration") == iter1:
                    iter1_data = entry
                if entry.get("iteration") == iter2:
                    iter2_data = entry
            
            if not iter1_data or not iter2_data:
                return {"error": "Iterations not found in history"}
            
            # Compare metrics
            comparison = {
                "iterations": [iter1, iter2],
                "metrics": {}
            }
            
            # Compare common metrics
            all_keys = set(iter1_data.keys()) | set(iter2_data.keys())
            for key in all_keys:
                # Skip non-numeric and metadata fields
                if key in ["iteration", "timestamp", "error", "domain_coverage"]:
                    continue
                
                # Get values
                val1 = iter1_data.get(key, 0.0)
                val2 = iter2_data.get(key, 0.0)
                
                # Skip if not numeric
                if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                    continue
                
                # Compute difference and percent change
                diff = val2 - val1
                pct_change = (diff / max(0.0001, abs(val1))) * 100 if val1 != 0 else float('inf')
                
                comparison["metrics"][key] = {
                    "values": [val1, val2],
                    "difference": diff,
                    "percent_change": pct_change
                }
            
            # Compare domain coverage separately
            if "domain_coverage" in iter1_data and "domain_coverage" in iter2_data:
                coverage1 = iter1_data["domain_coverage"]
                coverage2 = iter2_data["domain_coverage"]
                
                all_domains = set(coverage1.keys()) | set(coverage2.keys())
                domain_comparison = {}
                
                for domain in all_domains:
                    val1 = coverage1.get(domain, 0.0)
                    val2 = coverage2.get(domain, 0.0)
                    
                    diff = val2 - val1
                    pct_change = (diff / max(0.0001, abs(val1))) * 100 if val1 != 0 else float('inf')
                    
                    domain_comparison[domain] = {
                        "values": [val1, val2],
                        "difference": diff,
                        "percent_change": pct_change
                    }
                
                comparison["domain_coverage"] = domain_comparison
            
            return comparison
        except Exception as e:
            log.error(f"Failed to compare iterations: {e}")
            return {"error": str(e)}
    
    async def detect_anomalies(self, window_size: int = 3, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics history.
        
        Args:
            window_size: Number of previous iterations to consider
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List[Dict[str, Any]]: Detected anomalies
        """
        if len(self.history) < window_size + 1:
            return []
        
        try:
            # Get latest entry
            latest = self.history[-1]
            
            # Get previous window
            window = self.history[-(window_size+1):-1]
            
            # Detect anomalies
            anomalies = []
            
            # Check numeric metrics
            for key, value in latest.items():
                # Skip non-numeric and metadata fields
                if key in ["iteration", "timestamp", "error", "domain_coverage"]:
                    continue
                
                # Skip if not numeric
                if not isinstance(value, (int, float)):
                    continue
                
                # Get window values
                window_values = [entry.get(key, 0.0) for entry in window]
                
                # Compute mean and standard deviation
                mean = np.mean(window_values)
                std = np.std(window_values) or 1.0  # Avoid division by zero
                
                # Compute z-score
                z_score = abs((value - mean) / std)
                
                # Check if anomaly
                if z_score > threshold:
                    anomalies.append({
                        "metric": key,
                        "value": value,
                        "mean": mean,
                        "std": std,
                        "z_score": z_score,
                        "iteration": latest.get("iteration", 0)
                    })
            
            return anomalies
        except Exception as e:
            log.error(f"Failed to detect anomalies: {e}")
            return []
