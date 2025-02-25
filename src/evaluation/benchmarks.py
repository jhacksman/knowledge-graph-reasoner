"""Benchmarking module for knowledge graph reasoner."""
from typing import List, Dict, Any, Optional, Set, Tuple, AsyncIterator
import os
import logging
import json
import time
from pathlib import Path
import networkx as nx
import numpy as np
from datetime import datetime

from ..models.node import Node
from ..models.edge import Edge
from ..graph.manager import GraphManager
from ..reasoning.pipeline import ReasoningPipeline
from ..reasoning.llm import VeniceLLM
from .metrics import GraphEvaluator

log = logging.getLogger(__name__)


class GraphBenchmark:
    """Benchmarks for knowledge graph reasoning."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        reference_graphs_dir: Optional[str] = None
    ):
        """Initialize benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
            reference_graphs_dir: Directory containing reference graphs
        """
        self.output_dir = output_dir or "benchmarks"
        self.reference_graphs_dir = reference_graphs_dir or os.path.join(self.output_dir, "references")
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reference_graphs_dir, exist_ok=True)
        
        # Standard seed concepts for testing
        self.standard_seeds = {
            "science": [
                "quantum mechanics",
                "general relativity",
                "evolutionary biology",
                "neuroscience",
                "climate science"
            ],
            "technology": [
                "artificial intelligence",
                "blockchain",
                "quantum computing",
                "renewable energy",
                "biotechnology"
            ],
            "humanities": [
                "philosophy of mind",
                "cultural anthropology",
                "comparative literature",
                "economic history",
                "political theory"
            ],
            "interdisciplinary": [
                "cognitive science",
                "systems thinking",
                "complex systems",
                "information theory",
                "network science"
            ]
        }
    
    async def run_benchmark(
        self,
        pipeline: ReasoningPipeline,
        graph_manager: GraphManager,
        evaluator: GraphEvaluator,
        seed_concept: str,
        max_iterations: int = 10,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run benchmark with a seed concept.
        
        Args:
            pipeline: Reasoning pipeline
            graph_manager: Graph manager
            evaluator: Graph evaluator
            seed_concept: Seed concept
            max_iterations: Maximum iterations
            name: Benchmark name
            description: Benchmark description
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        try:
            # Set benchmark name
            if not name:
                timestamp = int(time.time())
                name = f"benchmark_{seed_concept.replace(' ', '_')}_{timestamp}"
            
            # Start time
            start_time = time.time()
            
            # Run pipeline
            log.info(f"Running benchmark with seed concept: {seed_concept}")
            
            # Store original max_iterations
            original_max_iterations = pipeline.max_iterations
            
            # Set max iterations for benchmark
            pipeline.max_iterations = max_iterations
            
            # Run pipeline
            final_state = await pipeline.expand_knowledge(seed_concept)
            
            # Restore original max_iterations
            pipeline.max_iterations = original_max_iterations
            
            # End time
            end_time = time.time()
            
            # Get evaluation results
            evaluation = await evaluator.evaluate(max_iterations)
            
            # Prepare benchmark results
            results = {
                "name": name,
                "description": description or f"Benchmark with seed concept: {seed_concept}",
                "seed_concept": seed_concept,
                "max_iterations": max_iterations,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": end_time - start_time,
                "final_state": final_state,
                "evaluation": evaluation
            }
            
            # Save results
            self._save_benchmark_results(results)
            
            return results
        except Exception as e:
            log.error(f"Failed to run benchmark: {e}")
            return {
                "name": name or "failed_benchmark",
                "error": str(e),
                "seed_concept": seed_concept,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_standard_benchmarks(
        self,
        pipeline: ReasoningPipeline,
        graph_manager: GraphManager,
        evaluator: GraphEvaluator,
        domain: str = "interdisciplinary",
        max_iterations: int = 10
    ) -> List[Dict[str, Any]]:
        """Run standard benchmarks for a domain.
        
        Args:
            pipeline: Reasoning pipeline
            graph_manager: Graph manager
            evaluator: Graph evaluator
            domain: Domain to benchmark
            max_iterations: Maximum iterations
            
        Returns:
            List[Dict[str, Any]]: Benchmark results
        """
        if domain not in self.standard_seeds:
            log.warning(f"Domain {domain} not found in standard seeds")
            return []
        
        results = []
        for seed in self.standard_seeds[domain]:
            # Run benchmark
            result = await self.run_benchmark(
                pipeline=pipeline,
                graph_manager=graph_manager,
                evaluator=evaluator,
                seed_concept=seed,
                max_iterations=max_iterations,
                name=f"standard_{domain}_{seed.replace(' ', '_')}",
                description=f"Standard {domain} benchmark with seed: {seed}"
            )
            
            results.append(result)
        
        return results
    
    async def compare_with_reference(
        self,
        graph_manager: GraphManager,
        reference_name: str
    ) -> Dict[str, Any]:
        """Compare current graph with a reference graph.
        
        Args:
            graph_manager: Graph manager
            reference_name: Name of reference graph
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            # Load reference graph
            reference = self._load_reference_graph(reference_name)
            if not reference:
                return {"error": f"Reference graph {reference_name} not found"}
            
            # Get current graph state
            current_state = await graph_manager.get_graph_state()
            
            # Create NetworkX graphs
            ref_graph: nx.Graph = nx.Graph()
            current_graph: nx.Graph = nx.Graph()
            
            # Add nodes and edges to reference graph
            for node in reference.get("nodes", []):
                ref_graph.add_node(node["id"], **node)
            
            for edge in reference.get("edges", []):
                ref_graph.add_edge(edge["source"], edge["target"], **edge)
            
            # Get all nodes and edges from current graph
            nodes = []
            for node in await self.graph_manager.get_all_concepts():
                nodes.append(node)
                current_graph.add_node(node.id, content=node.content, metadata=node.metadata)
            
            edges = []
            for edge in await self.graph_manager.get_all_relationships():
                edges.append(edge)
                current_graph.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.type,
                    metadata=edge.metadata
                )
            
            # Compare graphs
            comparison = {
                "reference_name": reference_name,
                "timestamp": datetime.now().isoformat(),
                "node_count": {
                    "reference": ref_graph.number_of_nodes(),
                    "current": current_graph.number_of_nodes(),
                    "difference": current_graph.number_of_nodes() - ref_graph.number_of_nodes()
                },
                "edge_count": {
                    "reference": ref_graph.number_of_edges(),
                    "current": current_graph.number_of_edges(),
                    "difference": current_graph.number_of_edges() - ref_graph.number_of_edges()
                }
            }
            
            # Compare graph metrics
            if nx.is_connected(ref_graph) and nx.is_connected(current_graph):
                # Average path length
                ref_path_length = nx.average_shortest_path_length(ref_graph)
                current_path_length = nx.average_shortest_path_length(current_graph)
                
                comparison["avg_path_length"] = {
                    "reference": ref_path_length,
                    "current": current_path_length,
                    "difference": current_path_length - ref_path_length
                }
                
                # Diameter
                ref_diameter = nx.diameter(ref_graph)
                current_diameter = nx.diameter(current_graph)
                
                comparison["diameter"] = {
                    "reference": ref_diameter,
                    "current": current_diameter,
                    "difference": current_diameter - ref_diameter
                }
            
            # Compare modularity
            try:
                ref_communities = nx.community.greedy_modularity_communities(ref_graph)
                current_communities = nx.community.greedy_modularity_communities(current_graph)
                
                ref_modularity = nx.community.modularity(ref_graph, ref_communities)
                current_modularity = nx.community.modularity(current_graph, current_communities)
                
                comparison["modularity"] = {
                    "reference": ref_modularity,
                    "current": current_modularity,
                    "difference": current_modularity - ref_modularity
                }
                
                comparison["community_count"] = {
                    "reference": len(ref_communities),
                    "current": len(current_communities),
                    "difference": len(current_communities) - len(ref_communities)
                }
            except Exception as e:
                log.warning(f"Failed to compare modularity: {e}")
            
            # Node overlap
            ref_nodes = set(ref_graph.nodes())
            current_nodes = set(current_graph.nodes())
            
            common_nodes = ref_nodes.intersection(current_nodes)
            
            comparison["node_overlap"] = {
                "common_count": len(common_nodes),
                "reference_only": len(ref_nodes - current_nodes),
                "current_only": len(current_nodes - ref_nodes),
                "overlap_percentage": len(common_nodes) / max(1, len(ref_nodes)) * 100
            }
            
            # Edge overlap
            ref_edges = set(ref_graph.edges())
            current_edges = set(current_graph.edges())
            
            common_edges = ref_edges.intersection(current_edges)
            
            comparison["edge_overlap"] = {
                "common_count": len(common_edges),
                "reference_only": len(ref_edges - current_edges),
                "current_only": len(current_edges - ref_edges),
                "overlap_percentage": len(common_edges) / max(1, len(ref_edges)) * 100
            }
            
            return comparison
        except Exception as e:
            log.error(f"Failed to compare with reference: {e}")
            return {"error": str(e)}
    
    async def create_reference_graph(
        self,
        graph_manager: GraphManager,
        name: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a reference graph from current state.
        
        Args:
            graph_manager: Graph manager
            name: Reference graph name
            description: Reference graph description
            
        Returns:
            Dict[str, Any]: Reference graph metadata
        """
        try:
            # Get current graph state
            state = await graph_manager.get_graph_state()
            
            # Get all nodes and edges
            nodes = []
            for node in await self.graph_manager.get_all_concepts():
                nodes.append({
                    "id": node.id,
                    "content": node.content,
                    "metadata": node.metadata
                })
            
            edges = []
            for edge in await self.graph_manager.get_all_relationships():
                edges.append({
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "metadata": edge.metadata
                })
            
            # Create reference graph
            reference = {
                "name": name,
                "description": description or f"Reference graph: {name}",
                "timestamp": datetime.now().isoformat(),
                "state": state,
                "nodes": nodes,
                "edges": edges
            }
            
            # Save reference graph
            self._save_reference_graph(reference)
            
            return {
                "name": name,
                "description": reference["description"],
                "timestamp": reference["timestamp"],
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        except Exception as e:
            log.error(f"Failed to create reference graph: {e}")
            return {"error": str(e)}
    
    def list_reference_graphs(self) -> List[Dict[str, Any]]:
        """List available reference graphs.
        
        Returns:
            List[Dict[str, Any]]: Reference graph metadata
        """
        try:
            references = []
            
            # List reference graph files
            for file_path in Path(self.reference_graphs_dir).glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        reference = json.load(f)
                    
                    references.append({
                        "name": reference.get("name", file_path.stem),
                        "description": reference.get("description", ""),
                        "timestamp": reference.get("timestamp", ""),
                        "node_count": len(reference.get("nodes", [])),
                        "edge_count": len(reference.get("edges", []))
                    })
                except Exception as e:
                    log.warning(f"Failed to load reference graph {file_path}: {e}")
            
            return references
        except Exception as e:
            log.error(f"Failed to list reference graphs: {e}")
            return []
    
    def list_benchmark_results(self) -> List[Dict[str, Any]]:
        """List available benchmark results.
        
        Returns:
            List[Dict[str, Any]]: Benchmark result metadata
        """
        try:
            results = []
            
            # List benchmark result files
            for file_path in Path(self.output_dir).glob("benchmark_*.json"):
                try:
                    with open(file_path, "r") as f:
                        benchmark = json.load(f)
                    
                    results.append({
                        "name": benchmark.get("name", file_path.stem),
                        "description": benchmark.get("description", ""),
                        "seed_concept": benchmark.get("seed_concept", ""),
                        "timestamp": benchmark.get("timestamp", ""),
                        "duration_seconds": benchmark.get("duration_seconds", 0),
                        "max_iterations": benchmark.get("max_iterations", 0)
                    })
                except Exception as e:
                    log.warning(f"Failed to load benchmark result {file_path}: {e}")
            
            return results
        except Exception as e:
            log.error(f"Failed to list benchmark results: {e}")
            return []
    
    def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file.
        
        Args:
            results: Benchmark results
        """
        try:
            # Create filename
            name = results.get("name", f"benchmark_{int(time.time())}")
            filename = os.path.join(self.output_dir, f"{name}.json")
            
            # Save to file
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            
            log.info(f"Saved benchmark results to {filename}")
        except Exception as e:
            log.error(f"Failed to save benchmark results: {e}")
    
    def _save_reference_graph(self, reference: Dict[str, Any]) -> None:
        """Save reference graph to file.
        
        Args:
            reference: Reference graph
        """
        try:
            # Create filename
            name = reference.get("name", f"reference_{int(time.time())}")
            filename = os.path.join(self.reference_graphs_dir, f"{name}.json")
            
            # Save to file
            with open(filename, "w") as f:
                json.dump(reference, f, indent=2)
            
            log.info(f"Saved reference graph to {filename}")
        except Exception as e:
            log.error(f"Failed to save reference graph: {e}")
    
    def _load_reference_graph(self, name: str) -> Optional[Dict[str, Any]]:
        """Load reference graph from file.
        
        Args:
            name: Reference graph name
            
        Returns:
            Optional[Dict[str, Any]]: Reference graph or None if not found
        """
        try:
            # Check if name includes .json extension
            if not name.endswith(".json"):
                name = f"{name}.json"
            
            # Create filename
            filename = os.path.join(self.reference_graphs_dir, name)
            
            # Check if file exists
            if not os.path.exists(filename):
                log.warning(f"Reference graph {filename} not found")
                return None
            
            # Load from file
            with open(filename, "r") as f:
                reference = json.load(f)
            
            return reference
        except Exception as e:
            log.error(f"Failed to load reference graph: {e}")
            return None
    
    def load_benchmark_result(self, name: str) -> Optional[Dict[str, Any]]:
        """Load benchmark result from file.
        
        Args:
            name: Benchmark result name
            
        Returns:
            Optional[Dict[str, Any]]: Benchmark result or None if not found
        """
        try:
            # Check if name includes .json extension
            if not name.endswith(".json"):
                name = f"{name}.json"
            
            # Create filename
            filename = os.path.join(self.output_dir, name)
            
            # Check if file exists
            if not os.path.exists(filename):
                log.warning(f"Benchmark result {filename} not found")
                return None
            
            # Load from file
            with open(filename, "r") as f:
                benchmark = json.load(f)
            
            return benchmark
        except Exception as e:
            log.error(f"Failed to load benchmark result: {e}")
            return None


class CustomBenchmark:
    """Custom benchmark definition."""
    
    def __init__(
        self,
        name: str,
        seed_concepts: List[str],
        max_iterations: int = 10,
        description: Optional[str] = None,
        target_metrics: Optional[Dict[str, float]] = None
    ):
        """Initialize custom benchmark.
        
        Args:
            name: Benchmark name
            seed_concepts: List of seed concepts
            max_iterations: Maximum iterations
            description: Benchmark description
            target_metrics: Target metrics for evaluation
        """
        self.name = name
        self.seed_concepts = seed_concepts
        self.max_iterations = max_iterations
        self.description = description or f"Custom benchmark: {name}"
        self.target_metrics = target_metrics or {}
    
    async def run(
        self,
        pipeline: ReasoningPipeline,
        graph_manager: GraphManager,
        evaluator: GraphEvaluator,
        benchmark_manager: GraphBenchmark
    ) -> Dict[str, Any]:
        """Run custom benchmark.
        
        Args:
            pipeline: Reasoning pipeline
            graph_manager: Graph manager
            evaluator: Graph evaluator
            benchmark_manager: Benchmark manager
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = []
        
        for seed in self.seed_concepts:
            # Run benchmark
            result = await benchmark_manager.run_benchmark(
                pipeline=pipeline,
                graph_manager=graph_manager,
                evaluator=evaluator,
                seed_concept=seed,
                max_iterations=self.max_iterations,
                name=f"{self.name}_{seed.replace(' ', '_')}",
                description=f"{self.description} - Seed: {seed}"
            )
            
            results.append(result)
        
        # Aggregate results
        aggregated = {
            "name": self.name,
            "description": self.description,
            "seed_concepts": self.seed_concepts,
            "max_iterations": self.max_iterations,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        # Compare with target metrics if provided
        if self.target_metrics:
            metric_comparison = {}
            
            for metric_name, target_value in self.target_metrics.items():
                actual_values = []
                
                for result in results:
                    if "evaluation" in result and metric_name in result["evaluation"]:
                        actual_values.append(result["evaluation"][metric_name])
                
                if actual_values:
                    avg_value = np.mean(actual_values)
                    metric_comparison[metric_name] = {
                        "target": target_value,
                        "actual": avg_value,
                        "difference": avg_value - target_value,
                        "achieved": abs(avg_value - target_value) / max(0.0001, abs(target_value)) <= 0.1  # Within 10%
                    }
            
            aggregated["metric_comparison"] = metric_comparison
        
        return aggregated
