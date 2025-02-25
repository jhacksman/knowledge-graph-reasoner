"""Compositional reasoning for knowledge graph expansion."""
from typing import List, Dict, Any, Optional
import logging
import asyncio

from .llm import VeniceLLM
from ..graph.manager import GraphManager
from ..models.node import Node
from .prompts import COMPOSITIONAL_REASONING_PROMPT

log = logging.getLogger(__name__)


class CompositionalReasoner:
    """Handles compositional reasoning over knowledge graph concepts."""

    def __init__(
        self,
        llm: VeniceLLM,
        graph: GraphManager
    ):
        """Initialize compositional reasoner.
        
        Args:
            llm: LLM interface
            graph: Graph manager
        """
        self.llm = llm
        self.graph = graph
    
    async def synthesize_concepts(
        self,
        concept_ids: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Synthesize higher-level concepts from existing ones.
        
        Args:
            concept_ids: List of concept IDs to synthesize
            max_depth: Maximum depth of compositional reasoning
            
        Returns:
            Dict[str, Any]: Results of compositional reasoning
        """
        try:
            # Get concepts from graph
            concepts = []
            for concept_id in concept_ids:
                node = await self.graph.get_concept(concept_id)
                if node:
                    concepts.append(node)
            
            if not concepts:
                log.warning("No valid concepts found for compositional reasoning")
                return {"success": False, "error": "No valid concepts found"}
            
            # Get relationships between concepts
            relationships = []
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    rels = await self.graph.get_relationships(
                        source_id=concept1.id,
                        target_id=concept2.id
                    )
                    relationships.extend(rels)
            
            # Perform iterative compositional reasoning
            results = await self._iterative_composition(concepts, relationships, max_depth)
            
            return {
                "success": True,
                "results": results
            }
            
        except Exception as e:
            log.error(f"Compositional reasoning failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _iterative_composition(
        self,
        concepts: List[Node],
        relationships: List[Any],  # Can be List[Dict[str, Any]] or List[Edge]
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Perform iterative compositional reasoning.
        
        Args:
            concepts: List of concept nodes
            relationships: List of relationships between concepts
            max_depth: Maximum depth of compositional reasoning
            
        Returns:
            List[Dict[str, Any]]: Results of compositional reasoning
        """
        results = []
        current_concepts = concepts
        current_relationships = relationships
        
        for depth in range(max_depth):
            log.info(f"Compositional reasoning depth {depth+1}/{max_depth}")
            
            # Format concepts and relationships for prompt
            concepts_text = "\n".join([
                f"- {node.id}: {node.content}" for node in current_concepts
            ])
            
            # Handle both dictionary-style relationships and Edge objects
            rel_text_items = []
            for rel in current_relationships:
                if hasattr(rel, 'source') and hasattr(rel, 'target') and hasattr(rel, 'type'):
                    # Edge object
                    rel_text_items.append(f"- {rel.source} -> {rel.target}: {rel.type}")
                elif isinstance(rel, dict) and 'source' in rel and 'target' in rel:
                    # Dictionary
                    rel_text_items.append(
                        f"- {rel['source']} -> {rel['target']}: {rel.get('type', 'related')}"
                    )
            
            relationships_text = "\n".join(rel_text_items)
            
            # Generate prompt
            prompt = COMPOSITIONAL_REASONING_PROMPT.format(
                concepts=concepts_text,
                relationships=relationships_text
            )
            
            # Get LLM response
            response = await self.llm.generate([{
                "role": "system",
                "content": "You are a knowledge graph reasoning assistant."
            }, {
                "role": "user",
                "content": prompt
            }])
            
            # Process response
            try:
                content = response["choices"][0]["message"]["content"]
                
                # Parse entities and relationships using the parser
                # This would typically use the EntityRelationshipParser from extraction module
                # For simplicity, we'll just store the raw response for now
                
                result = {
                    "depth": depth + 1,
                    "content": content,
                    "source_concepts": [node.id for node in current_concepts],
                    "source_relationships": current_relationships
                }
                
                results.append(result)
                
                # For next iteration, we would extract new concepts and relationships
                # from the response and use them as input for the next level
                # This is a simplified version
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                log.error(f"Error in compositional reasoning iteration {depth+1}: {e}")
                break
        
        return results
    
    async def analyze_path(
        self,
        path_nodes: List[str]
    ) -> Dict[str, Any]:
        """Analyze a path in the knowledge graph.
        
        Args:
            path_nodes: List of node IDs forming a path
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Get nodes from graph
            nodes = []
            for node_id in path_nodes:
                node = await self.graph.get_concept(node_id)
                if node:
                    nodes.append(node)
            
            if len(nodes) < 2:
                return {"success": False, "error": "Path too short for analysis"}
            
            # Get relationships along the path
            relationships = []
            for i in range(len(nodes) - 1):
                rels = await self.graph.get_relationships(
                    source_id=nodes[i].id,
                    target_id=nodes[i+1].id
                )
                relationships.extend(rels)
            
            # Perform compositional reasoning on the path
            results = await self._iterative_composition(nodes, relationships, max_depth=1)
            
            return {
                "success": True,
                "path_analysis": results[0] if results else None,
                "path_length": len(nodes),
                "node_count": len(nodes),
                "relationship_count": len(relationships)
            }
            
        except Exception as e:
            log.error(f"Path analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
