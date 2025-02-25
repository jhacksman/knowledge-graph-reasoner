"""Materials design example application for knowledge graph reasoner.

This example demonstrates how the knowledge graph reasoner can be used
to design impact-resistant materials through iterative reasoning.
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


class MaterialsDesignExample:
    """Materials design example application.
    
    Demonstrates how the knowledge graph reasoner can be used to design
    impact-resistant materials through iterative reasoning.
    """
    
    def __init__(
        self,
        initial_prompt: str = "Describe a way to design impact resistant materials",
        max_iterations: int = 10,
        output_dir: Optional[str] = None
    ):
        """Initialize materials design example.
        
        Args:
            initial_prompt: Initial prompt to start the reasoning process
            max_iterations: Maximum number of reasoning iterations
            output_dir: Directory to save outputs (default: None)
        """
        self.initial_prompt = initial_prompt
        self.max_iterations = max_iterations
        self.output_dir = output_dir or os.path.join(os.getcwd(), "outputs", "materials_design")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.llm = create_venice_llm(rate_limited=True)
        self.pipeline = ReasoningPipeline(self.llm)
        self.parser = EntityRelationshipParser()
        self.bridge_integration = BridgeNodeIntegration()
        self.analytics = AdvancedAnalytics()
        
        # Initialize graph
        self.graph: nx.Graph = nx.Graph()
        
        # Domain mapping for materials science
        self.domain_mapping: Dict[str, str] = {}
        
        # Track iterations
        self.iterations: List[Dict[str, Any]] = []
    
    async def initialize(self) -> None:
        """Initialize the example."""
        # Set up the pipeline
        self.pipeline.set_max_iterations(self.max_iterations)
        
        # Set up the graph
        self.bridge_integration.set_graph(self.graph)
        self.analytics.set_graph(self.graph)
    
    async def run(self) -> Dict[str, Any]:
        """Run the materials design example.
        
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
                response = await self.pipeline.generate_response(current_prompt)
                
                # Parse entities and relationships
                parsed_data = await self.parser.parse(response)
                
                # Update graph with new concepts and relationships
                await self._update_graph(parsed_data)
                
                # Update domain mapping
                await self._update_domain_mapping(parsed_data)
                
                # Track bridge nodes
                bridge_summary = await self.bridge_integration.update_bridge_nodes()
                
                # Track analytics
                analytics_metrics = await self.analytics.track_metrics(snapshot_graph=True)
                
                # Generate next prompt based on graph state
                next_prompt = await self._generate_next_prompt(
                    response, parsed_data, bridge_summary, analytics_metrics
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
                convergence = await self.analytics.check_convergence()
                if all(convergence.values()):
                    log.info(f"Converged after {i+1} iterations")
                    break
            
            # Generate final summary
            summary = await self._generate_summary()
            
            # Save summary to file
            self._save_summary(summary)
            
            return summary
        except Exception as e:
            log.error(f"Error running materials design example: {e}")
            return {
                "status": "error",
                "error": str(e),
                "iterations": len(self.iterations)
            }
    
    async def _update_graph(self, parsed_data: Dict[str, Any]) -> None:
        """Update graph with parsed data.
        
        Args:
            parsed_data: Parsed data from LLM response
        """
        try:
            # Add concepts as nodes
            for concept in parsed_data.get("concepts", []):
                self.graph.add_node(
                    concept["name"],
                    type="concept",
                    description=concept.get("description", ""),
                    properties=concept.get("properties", {})
                )
            
            # Add relationships as edges
            for relationship in parsed_data.get("relationships", []):
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
    
    async def _update_domain_mapping(self, parsed_data: Dict[str, Any]) -> None:
        """Update domain mapping with parsed data.
        
        Args:
            parsed_data: Parsed data from LLM response
        """
        try:
            # Map concepts to domains
            for concept in parsed_data.get("concepts", []):
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
                elif any(kw in description for kw in ["polymer", "plastic", "elastomer"]):
                    domain = "polymers"
                elif any(kw in description for kw in ["metal", "alloy", "steel"]):
                    domain = "metals"
                elif any(kw in description for kw in ["ceramic", "glass", "oxide"]):
                    domain = "ceramics"
                elif any(kw in description for kw in ["composite", "fiber", "matrix"]):
                    domain = "composites"
                elif any(kw in description for kw in ["structure", "lattice", "crystal"]):
                    domain = "material_structure"
                elif any(kw in description for kw in ["property", "strength", "toughness"]):
                    domain = "material_properties"
                elif any(kw in description for kw in ["test", "characterization", "analysis"]):
                    domain = "testing_methods"
                elif any(kw in description for kw in ["process", "manufacturing", "synthesis"]):
                    domain = "manufacturing_processes"
                elif any(kw in description for kw in ["application", "use", "industry"]):
                    domain = "applications"
                else:
                    # Default domain
                    domain = "general_materials_science"
                
                # Update domain mapping
                self.domain_mapping[name] = domain
            
            # Update bridge integration with domain mapping
            self.bridge_integration.set_domain_mapping(self.domain_mapping)
        except Exception as e:
            log.error(f"Error updating domain mapping: {e}")
    
    async def _generate_next_prompt(
        self,
        response: str,
        parsed_data: Dict[str, Any],
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
            # Get high influence bridge nodes
            high_influence_nodes = bridge_summary.get("high_influence_nodes", [])
            
            # Get cross-domain bridge nodes
            cross_domain_nodes = bridge_summary.get("cross_domain_nodes", [])
            
            # Get bridge node recommendations
            recommendations = bridge_summary.get("recommendations", [])
            
            # Get concepts from parsed data
            concepts = parsed_data.get("concepts", [])
            
            # Build prompt based on graph state
            prompt_parts = []
            
            # Add base instruction
            prompt_parts.append("Based on our discussion about impact-resistant materials, I'd like you to explore further.")
            
            # Add focus on high influence nodes
            if high_influence_nodes:
                node_names = ", ".join(high_influence_nodes[:3])
                prompt_parts.append(f"Particularly, elaborate on the role of {node_names} in impact resistance.")
            
            # Add focus on cross-domain connections
            if cross_domain_nodes:
                node_names = ", ".join(cross_domain_nodes[:2])
                prompt_parts.append(f"Also, explain how {node_names} connect different aspects of materials science.")
            
            # Add focus on recommended connections
            if recommendations:
                rec = recommendations[0]
                source = rec.get("source")
                target = rec.get("target")
                if source and target:
                    prompt_parts.append(f"Consider the potential relationship between {source} and {target}.")
            
            # Add focus on unexplored concepts
            if concepts:
                # Find concepts with minimal description
                unexplored = [c["name"] for c in concepts if len(c.get("description", "")) < 50]
                if unexplored:
                    concept_name = unexplored[0]
                    prompt_parts.append(f"Provide more details about {concept_name} and its properties.")
            
            # Add request for specific material examples
            prompt_parts.append("Can you provide specific examples of materials or material combinations that demonstrate these principles?")
            
            # Add request for manufacturing considerations
            prompt_parts.append("What manufacturing processes would be most suitable for these materials?")
            
            # Combine prompt parts
            next_prompt = " ".join(prompt_parts)
            
            return next_prompt
        except Exception as e:
            log.error(f"Error generating next prompt: {e}")
            return "Continue exploring impact-resistant materials. What other approaches or materials should we consider?"
    
    async def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of the materials design example.
        
        Returns:
            Dict[str, Any]: Summary
        """
        try:
            # Get graph statistics
            num_nodes = len(self.graph.nodes())
            num_edges = len(self.graph.edges())
            
            # Get domain statistics
            domains = set(self.domain_mapping.values())
            domain_counts = {}
            for domain in domains:
                domain_counts[domain] = sum(1 for d in self.domain_mapping.values() if d == domain)
            
            # Get bridge node summary
            bridge_summary = await self.bridge_integration.get_bridge_metrics()
            
            # Get analytics summary
            analytics_summary = await self.analytics.get_summary_report()
            
            # Extract key concepts
            centrality = nx.degree_centrality(self.graph)
            key_concepts = sorted(
                centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Extract key relationships
            key_relationships = []
            for source, target, data in self.graph.edges(data=True):
                if "type" in data:
                    key_relationships.append({
                        "source": source,
                        "target": target,
                        "type": data["type"],
                        "description": data.get("description", "")
                    })
            
            # Sort by source centrality
            key_relationships = sorted(
                key_relationships,
                key=lambda x: centrality.get(x["source"], 0),
                reverse=True
            )[:15]
            
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
                "key_concepts": [
                    {"name": name, "centrality": score}
                    for name, score in key_concepts
                ],
                "key_relationships": key_relationships,
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


async def run_materials_design_example(
    initial_prompt: str = "Describe a way to design impact resistant materials",
    max_iterations: int = 10,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run the materials design example.
    
    Args:
        initial_prompt: Initial prompt to start the reasoning process
        max_iterations: Maximum number of reasoning iterations
        output_dir: Directory to save outputs (default: None)
        
    Returns:
        Dict[str, Any]: Results of the example
    """
    example = MaterialsDesignExample(
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
    asyncio.run(run_materials_design_example())
