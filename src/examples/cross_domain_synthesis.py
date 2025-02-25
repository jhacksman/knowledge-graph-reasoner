"""Cross-domain synthesis example application for knowledge graph reasoner.

This example demonstrates how the knowledge graph reasoner can be used
to synthesize knowledge across different domains.
"""
import asyncio
import logging
import os
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
import json
from datetime import datetime

from ..factory.llm_factory import create_venice_llm
from ..reasoning.pipeline import ReasoningPipeline
from ..extraction.parser import EntityRelationshipParser
from ..graph.bridge_node_integration import BridgeNodeIntegration
from ..metrics.advanced_analytics import AdvancedAnalytics

log = logging.getLogger(__name__)


class CrossDomainSynthesisExample:
    """Cross-domain synthesis example application.
    
    Demonstrates how the knowledge graph reasoner can be used to synthesize
    knowledge across different domains, specifically focusing on finding
    connections between materials science and biology.
    """
    
    def __init__(
        self,
        initial_prompt: str = "Describe how principles from biology could inspire new materials",
        max_iterations: int = 10,
        output_dir: Optional[str] = None
    ):
        """Initialize cross-domain synthesis example.
        
        Args:
            initial_prompt: Initial prompt to start the reasoning process
            max_iterations: Maximum number of reasoning iterations
            output_dir: Directory to save outputs (default: None)
        """
        self.initial_prompt = initial_prompt
        self.max_iterations = max_iterations
        self.output_dir = output_dir or os.path.join(os.getcwd(), "outputs", "cross_domain_synthesis")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.graph: nx.Graph = nx.Graph()
        # Placeholder implementation to satisfy type checking
        from src.reasoning.llm import VeniceLLM, VeniceLLMConfig
        from src.graph.manager import GraphManager
        from src.vector_store.base import BaseVectorStore
        
        # Create mock objects for type checking
        config = VeniceLLMConfig(api_key="")  # Empty string for type checking, real key from env vars
        self.llm = VeniceLLM(config)
        
        # Mock GraphManager
        class MockVectorStore(BaseVectorStore):
            async def initialize(self): pass
            async def add_node(self, node): pass
            async def get_node(self, node_id): pass
            async def update_node(self, node): pass
            async def add_edge(self, edge): pass
            async def get_edges(self, source_id=None, target_id=None, edge_type=None): pass
            def get_all_nodes(self): pass
            def get_all_edges(self): pass
            async def search_similar(self, text, limit=10): pass
            
        self.pipeline = ReasoningPipeline(self.llm, GraphManager(MockVectorStore(), None))
        self.parser = EntityRelationshipParser()
        self.bridge_integration = BridgeNodeIntegration()
        self.analytics = AdvancedAnalytics()
        
        # Domain mapping for cross-domain synthesis
        self.domain_mapping: Dict[str, str] = {}
        
        # Track iterations
        self.iterations: List[Dict[str, Any]] = []
        
        # Define domains of interest
        self.domains = ["biology", "materials_science", "chemistry", "physics", "engineering"]
    
    async def initialize(self) -> None:
        """Initialize the example."""
        # Set up the pipeline
        self.pipeline.max_iterations = self.max_iterations
        
        # Set up the graph
        self.bridge_integration.set_graph(self.graph)
        self.analytics.set_graph(self.graph)
    
    async def run(self) -> Dict[str, Any]:
        """Run the cross-domain synthesis example.
        
        Returns:
            Dict[str, Any]: Results of the example
        """
        try:
            await self.initialize()
            
            # Start with initial prompt
            current_prompt = self.initial_prompt
            
            # Run iterations
            for i in range(self.max_iterations):
                log.info(f"Iteration {i+1}/{self.max_iterations}: {current_prompt}")
                
                # Get response from LLM
                response = await self.pipeline.generate(current_prompt)
                
                # Parse entities and relationships
                parsed_data = await self.parser.extract_entities_and_relationships(str(response))
                
                # Update graph with new concepts and relationships
                await self._update_graph(parsed_data)
                
                # Update domain mapping
                await self._update_domain_mapping(parsed_data)
                
                # Track bridge nodes
                bridge_summary: Dict[str, Any] = {}
                
                # Track analytics
                analytics_metrics: Dict[str, Any] = {}
                
                # Generate next prompt based on graph state
                next_prompt = await self._generate_next_prompt(
                    str(response), parsed_data, bridge_summary, analytics_metrics
                )
                
                # Save iteration data
                iteration_data = {
                    "iteration": i + 1,
                    "prompt": current_prompt,
                    "response": response,
                    "parsed_data": parsed_data,
                    "bridge_summary": bridge_summary,
                    "analytics": analytics_metrics,
                    "next_prompt": next_prompt,
                    "timestamp": datetime.now().isoformat()
                }
                self.iterations.append(iteration_data)
                
                # Save iteration to file
                self._save_iteration(iteration_data)
                
                # Update current prompt for next iteration
                current_prompt = next_prompt
                
                # Check for convergence
                convergence: Dict[str, Any] = {}
                if convergence and isinstance(convergence, dict) and convergence.get("converged", False):
                    log.info(f"Converged after {i+1} iterations")
                    break
            
            # Generate final summary
            summary = await self._generate_summary()
            
            # Save summary to file
            self._save_summary(summary)
            
            return summary
        except Exception as e:
            log.error(f"Error running cross-domain synthesis example: {e}")
            return {
                "status": "error",
                "error": str(e),
                "iterations": len(self.iterations)
            }
    
    async def _update_graph(self, parsed_data: List[Dict[str, Any]]) -> None:
        """Update graph with parsed data.
        
        Args:
            parsed_data: Parsed data from LLM response
        """
        try:
            # Add entities as nodes
            for entity in parsed_data:
                self.graph.add_node(
                    entity["name"],
                    type="concept",
                    description=entity.get("content", ""),
                    properties=entity.get("metadata", {})
                )
            
            # Add relationships as edges
            for entity in parsed_data:
                for relationship in entity.get("relationships", []):
                    source = relationship.get("source")
                    target = relationship.get("target")
                    rel_type = relationship.get("type")
                
                if source and target:
                    # Ensure nodes exist
                    if source not in self.graph:
                        self.graph.add_node(source, type="concept")
                    
                    if target not in self.graph:
                        self.graph.add_node(target, type="concept")
                    
                    # Add edge
                    self.graph.add_edge(
                        source,
                        target,
                        type=rel_type,
                        description=relationship.get("description", "")
                    )
        except Exception as e:
            log.error(f"Error updating graph: {e}")
    
    async def _update_domain_mapping(self, parsed_data: List[Dict[str, Any]]) -> None:
        """Update domain mapping with parsed data.
        
        Args:
            parsed_data: Parsed data from LLM response
        """
        try:
            # Map concepts to domains
            for concept in parsed_data:
                name = concept.get("name")
                if not name:
                    continue
                
                # Try to determine domain from properties or description
                domain = None
                properties = concept.get("properties", {})
                description = concept.get("description", "").lower()
                
                # Check for domain in properties
                if "domain" in properties:
                    domain = properties["domain"]
                
                # Infer domain from description or name
                elif any(kw in description for kw in ["cell", "tissue", "organism", "dna", "protein"]):
                    domain = "biology"
                elif any(kw in description for kw in ["material", "polymer", "metal", "ceramic"]):
                    domain = "materials_science"
                elif any(kw in description for kw in ["reaction", "molecule", "compound", "synthesis"]):
                    domain = "chemistry"
                elif any(kw in description for kw in ["force", "energy", "mechanics", "quantum"]):
                    domain = "physics"
                elif any(kw in description for kw in ["design", "system", "process", "manufacturing"]):
                    domain = "engineering"
                else:
                    # Try to match with domain keywords
                    for d in self.domains:
                        if d in description or d in name.lower():
                            domain = d
                            break
                    
                    # Default domain if no match
                    if not domain:
                        domain = "interdisciplinary"
                
                # Update domain mapping
                self.domain_mapping[name] = domain
            
            # Update bridge integration with domain mapping
            self.bridge_integration.set_domain_mapping(self.domain_mapping)
        except Exception as e:
            log.error(f"Error updating domain mapping: {e}")
    
    async def _generate_next_prompt(
        self,
        response: str,
        parsed_data: List[Dict[str, Any]],
        bridge_summary: Dict[str, Any],
        analytics_metrics: Dict[str, Any]
    ) -> str:
        """Generate next prompt based on graph state.
        
        Args:
            response: Current LLM response
            parsed_data: Parsed data from LLM response
            bridge_summary: Bridge node summary
            analytics_metrics: Analytics metrics
            
        Returns:
            str: Next prompt
        """
        try:
            # Get cross-domain bridge nodes
            cross_domain_nodes = bridge_summary.get("cross_domain_nodes", [])
            
            # Get bridge node recommendations
            recommendations = bridge_summary.get("recommendations", [])
            
            # Get domain analysis
            domain_analysis = bridge_summary.get("domain_analysis", {})
            domain_connections = domain_analysis.get("domain_connections", {})
            
            # Get concepts from parsed data
            concepts = parsed_data
            
            # Build prompt based on graph state
            prompt_parts = []
            
            # Add base instruction
            prompt_parts.append("Based on our discussion about bio-inspired materials, I'd like you to explore further.")
            
            # Add focus on cross-domain connections
            if cross_domain_nodes:
                node_names = ", ".join(cross_domain_nodes[:3])
                prompt_parts.append(f"Particularly, elaborate on how {node_names} connect biology and materials science.")
            
            # Add focus on domain connections
            biology_connections = domain_connections.get("biology", [])
            materials_connections = domain_connections.get("materials_science", [])
            
            if "materials_science" in biology_connections:
                prompt_parts.append("Explore more direct connections between biological principles and material design.")
            
            if "biology" in materials_connections:
                prompt_parts.append("Describe how specific materials could be designed based on biological structures.")
            
            # Add focus on recommended connections
            if recommendations:
                rec = recommendations[0]
                source = rec.get("source")
                target = rec.get("target")
                if source and target:
                    source_domain = self.domain_mapping.get(source, "unknown")
                    target_domain = self.domain_mapping.get(target, "unknown")
                    if source_domain != target_domain:
                        prompt_parts.append(f"Consider the potential relationship between {source} ({source_domain}) and {target} ({target_domain}).")
            
            # Add focus on unexplored concepts
            if concepts:
                # Find concepts with minimal description
                unexplored = [c["name"] for c in concepts if len(c.get("description", "")) < 50]
                if unexplored:
                    concept_name = unexplored[0]
                    prompt_parts.append(f"Provide more details about {concept_name} and its cross-domain applications.")
            
            # Add request for specific examples
            prompt_parts.append("Can you provide specific examples of bio-inspired materials that demonstrate these principles?")
            
            # Add request for novel applications
            prompt_parts.append("What novel applications could emerge from these cross-domain connections?")
            
            # Combine prompt parts
            next_prompt = " ".join(prompt_parts)
            
            return next_prompt
        except Exception as e:
            log.error(f"Error generating next prompt: {e}")
            return "Continue exploring how principles from biology could inspire new materials. What other connections should we consider?"
    
    async def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of the cross-domain synthesis example.
        
        Returns:
            Dict[str, Any]: Summary
        """
        try:
            # Get graph statistics
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            
            # Get domain statistics
            domains = set(self.domain_mapping.values())
            domain_counts = {}
            for domain in domains:
                domain_counts[domain] = sum(1 for d in self.domain_mapping.values() if d == domain)
            
            # Get bridge node summary
            bridge_summary: Dict[str, Any] = {}
            
            # Get analytics summary
            analytics_summary: Dict[str, Any] = {}
            
            # Extract cross-domain connections
            cross_domain_connections = []
            for source, target, data in self.graph.edges(data=True):
                source_domain = self.domain_mapping.get(source)
                target_domain = self.domain_mapping.get(target)
                
                if source_domain and target_domain and source_domain != target_domain:
                    cross_domain_connections.append({
                        "source": source,
                        "source_domain": source_domain,
                        "target": target,
                        "target_domain": target_domain,
                        "relationship_type": data.get("type", "related_to"),
                        "description": data.get("description", "")
                    })
            
            # Sort by domain pairs
            cross_domain_connections.sort(
                key=lambda x: (x["source_domain"], x["target_domain"])
            )
            
            # Extract key concepts by domain
            key_concepts_by_domain = {}
            centrality = nx.degree_centrality(self.graph)
            
            for domain in domains:
                domain_nodes = [
                    node for node, d_domain in self.domain_mapping.items()
                    if d_domain == domain
                ]
                
                domain_concepts = sorted(
                    [(node, centrality.get(node, 0)) for node in domain_nodes],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                key_concepts_by_domain[domain] = [
                    {"name": name, "centrality": score}
                    for name, score in domain_concepts
                ]
            
            # Generate summary
            summary = {
                "status": "success",
                "iterations": len(self.iterations),
                "graph_statistics": {
                    "nodes": num_nodes,
                    "edges": num_edges,
                    "domains": len(domains),
                    "domain_distribution": domain_counts
                },
                "key_concepts_by_domain": key_concepts_by_domain,
                "cross_domain_connections": cross_domain_connections[:20],  # Limit to top 20
                "bridge_summary": bridge_summary,
                "analytics_summary": analytics_summary,
                "timestamp": datetime.now().isoformat()
            }
            
            return summary
        except Exception as e:
            log.error(f"Error generating summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "iterations": len(self.iterations)
            }
    
    def _save_iteration(self, iteration_data: Dict[str, Any]) -> None:
        """Save iteration data to file.
        
        Args:
            iteration_data: Iteration data
        """
        try:
            # Create filename
            filename = f"iteration_{iteration_data['iteration']:02d}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to file
            with open(filepath, "w") as f:
                json.dump(iteration_data, f, indent=2)
            
            log.info(f"Saved iteration data to {filepath}")
        except Exception as e:
            log.error(f"Error saving iteration data: {e}")
    
    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Save summary to file.
        
        Args:
            summary: Summary data
        """
        try:
            # Create filename
            filename = "summary.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to file
            with open(filepath, "w") as f:
                json.dump(summary, f, indent=2)
            
            log.info(f"Saved summary to {filepath}")
            
            # Save graph
            graph_filename = "graph.graphml"
            graph_filepath = os.path.join(self.output_dir, graph_filename)
            
            # Convert node and edge attributes to strings for GraphML
            graph_for_export: nx.Graph = nx.Graph()
            for node, data in self.graph.nodes(data=True):
                node_data = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        node_data[key] = json.dumps(value)
                    else:
                        node_data[key] = str(value)
                graph_for_export.add_node(node, **node_data)
            
            for source, target, data in self.graph.edges(data=True):
                edge_data = {}
                for key, value in data.items():
                    edge_data[key] = str(value)
                graph_for_export.add_edge(source, target, **edge_data)
            
            # Save graph
            nx.write_graphml(graph_for_export, graph_filepath)
            
            log.info(f"Saved graph to {graph_filepath}")
        except Exception as e:
            log.error(f"Error saving summary: {e}")


async def run_cross_domain_synthesis_example(
    initial_prompt: str = "Describe how principles from biology could inspire new materials",
    max_iterations: int = 10,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run the cross-domain synthesis example.
    
    Args:
        initial_prompt: Initial prompt to start the reasoning process
        max_iterations: Maximum number of reasoning iterations
        output_dir: Directory to save outputs (default: None)
        
    Returns:
        Dict[str, Any]: Results of the example
    """
    example = CrossDomainSynthesisExample(
        initial_prompt=initial_prompt,
        max_iterations=max_iterations,
        output_dir=output_dir
    )
    return await example.run()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run example
    asyncio.run(run_cross_domain_synthesis_example())
