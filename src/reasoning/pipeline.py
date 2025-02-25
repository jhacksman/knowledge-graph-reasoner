"""Reasoning pipeline implementation."""
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
import random

from .llm import VeniceLLM
from .compositional import CompositionalReasoner
from .prompts import ITERATIVE_REASONING_PROMPT
from ..graph.manager import GraphManager
from ..models.node import Node
from ..extraction.parser import EntityRelationshipParser
from ..extraction.deduplication import DeduplicationHandler

log = logging.getLogger(__name__)


class ReasoningPipeline:
    """Implements iterative graph reasoning pipeline."""
    
    def __init__(
        self,
        llm: VeniceLLM,
        graph: GraphManager,
        max_iterations: int = 100,
        stability_window: int = 5,
        min_path_length: float = 4.5,
        max_path_length: float = 5.0,
        min_diameter: float = 16.0,
        max_diameter: float = 18.0,
        similarity_threshold: float = 0.85,
        enable_compositional: bool = True
    ):
        """Initialize pipeline.
        
        Args:
            llm: LLM client for reasoning
            graph: Graph manager
            max_iterations: Maximum iterations
            stability_window: Window size for stability check
            min_path_length: Target minimum average path length
            max_path_length: Target maximum average path length
            min_diameter: Target minimum graph diameter
            max_diameter: Target maximum graph diameter
            similarity_threshold: Threshold for entity deduplication
            enable_compositional: Enable compositional reasoning
        """
        self.llm = llm
        self.graph = graph
        self.max_iterations = max_iterations
        self.stability_window = stability_window
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        
        # Add extraction components
        self.parser = EntityRelationshipParser()
        self.deduplication = DeduplicationHandler(similarity_threshold)
        
        # Add compositional reasoning
        self.enable_compositional = enable_compositional
        if enable_compositional:
            self.compositional = CompositionalReasoner(llm, graph)
        
        # Track metrics history
        self.metric_history: List[Dict[str, Any]] = []
    
    async def expand_knowledge(
        self,
        seed_concept: str,
        context: Optional[Dict[str, Any]] = None,
        compositional_interval: int = 10
    ) -> Dict[str, Any]:
        """Iteratively expand knowledge graph.

        Args:
            seed_concept: Initial concept
            context: Optional context for reasoning
            compositional_interval: Interval for compositional reasoning

        Returns:
            Dict[str, Any]: Final graph state
        """
        try:
            # Initialize with seed concept
            embedding = await self.llm.embed_text(seed_concept)
            seed_id = await self.graph.add_concept(seed_concept, embedding)

            # Track focus concept
            focus_concept = seed_concept
            focus_id = seed_id

            # Track significant nodes for compositional reasoning
            significant_nodes: List[str] = [seed_id]
            
            # Iterative expansion
            for i in range(self.max_iterations):
                log.info(f"Starting iteration {i+1}")

                # Get current graph state
                state = await self.graph.get_graph_state()
                self.metric_history.append(state)

                # Perform compositional reasoning at intervals if enabled
                if self.enable_compositional and i > 0 and i % compositional_interval == 0:
                    log.info(f"Performing compositional reasoning at iteration {i+1}")
                    await self._perform_compositional_reasoning(significant_nodes)

                # Generate new concepts
                concepts = await self._generate_concepts(
                    focus_concept,
                    state,
                    context
                )

                # Add new concepts and relationships
                new_nodes = await self._integrate_concepts(concepts)

                # Update significant nodes list with high-centrality nodes
                if new_nodes and len(new_nodes) > 0:
                    # Add nodes with high centrality or bridge nodes
                    for node in new_nodes:
                        if node.id in state.get("bridge_nodes", []):
                            significant_nodes.append(node.id)
                    
                    # Keep list at reasonable size
                    if len(significant_nodes) > 20:
                        significant_nodes = significant_nodes[-20:]

                # Update focus concept if new nodes were added
                if new_nodes and len(new_nodes) > 0:
                    # Choose a new focus concept from bridge nodes or recent additions
                    if state.get("bridge_nodes") and len(state["bridge_nodes"]) > 0:
                        # 50% chance to focus on a bridge node
                        if random.random() < 0.5:
                            bridge_id = state["bridge_nodes"][0]
                            bridge_node = await self.graph.get_concept(bridge_id)
                            if bridge_node:
                                focus_concept = bridge_node.content
                                focus_id = bridge_node.id
                        else:
                            # Focus on a new node
                            focus_concept = new_nodes[0].content
                            focus_id = new_nodes[0].id
                    else:
                        # Focus on a new node
                        focus_concept = new_nodes[0].content
                        focus_id = new_nodes[0].id

                # Check stability
                if await self._check_stability():
                    log.info("Graph reached stable state")
                    break

                # Rate limiting
                await asyncio.sleep(1)

            # Perform final compositional reasoning if enabled
            if self.enable_compositional:
                log.info("Performing final compositional reasoning")
                await self._perform_compositional_reasoning(significant_nodes)

            # Get final state
            return await self.graph.get_graph_state()

        except Exception as e:
            log.error(f"Knowledge expansion failed: {e}")
            raise
    
    async def _generate_concepts(
        self,
        seed_concept: str,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate new concepts using LLM.
        
        Args:
            seed_concept: Seed concept
            state: Current graph state
            context: Optional context
            
        Returns:
            List[Dict[str, Any]]: New concepts with metadata
        """
        # Construct prompt with structured output format
        prompt = self._build_generation_prompt(
            seed_concept,
            state,
            context
        )
        
        # Add instructions for structured output
        prompt += "\n\nFormat your response with entities and relationships in this format:\n"
        prompt += "<entity>entity_name: entity_description</entity>\n"
        prompt += "<relationship>source_entity: target_entity: relationship_type: description</relationship>"
        
        # Get LLM response
        response = await self.llm.generate([{
            "role": "system",
            "content": "You are a knowledge graph reasoning system."
        }, {
            "role": "user",
            "content": prompt
        }])
        
        # Parse response using the entity-relationship parser
        try:
            content = response["choices"][0]["message"]["content"]
            entities, relationships = self.parser.parse_response(content)
            
            # Process entities and get embeddings
            concepts = []
            for entity in entities:
                concepts.append({
                    "name": entity["name"],
                    "content": entity["content"],
                    "metadata": entity["metadata"],
                    "relationships": [
                        rel for rel in relationships 
                        if rel["source"] == entity["name"] or rel["target"] == entity["name"]
                    ]
                })
            
            return concepts
        except Exception as e:
            log.error(f"Failed to parse LLM response: {e}")
            return []
    
    async def _integrate_concepts(
        self,
        concepts: List[Dict[str, Any]]
    ) -> List[Node]:
        """Integrate new concepts into graph.
        
        Args:
            concepts: New concepts to integrate
            
        Returns:
            List[Node]: List of newly added nodes
        """
        # Get embeddings for all concepts
        embeddings = {}
        for concept in concepts:
            try:
                embedding = await self.llm.embed_text(concept["content"])
                embeddings[concept["name"]] = embedding
            except Exception as e:
                log.error(f"Failed to get embedding for concept {concept['name']}: {e}")
        
        # Get existing nodes for deduplication
        existing_nodes = []
        for concept in concepts:
            if concept["name"] in embeddings:
                similar = await self.graph.get_similar_concepts(
                    embeddings[concept["name"]],
                    k=5,
                    threshold=0.7
                )
                existing_nodes.extend(similar)
        
        # Find duplicates
        duplicates = self.deduplication.find_duplicates(
            concepts,
            existing_nodes,
            embeddings
        )
        
        # Add new concepts and track added nodes
        added_nodes = []
        concept_id_map = {}  # Map concept names to IDs
        
        for concept in concepts:
            try:
                # Skip if duplicate
                if concept["name"] in duplicates:
                    concept_id_map[concept["name"]] = duplicates[concept["name"]]
                    continue
                
                # Get embedding
                embedding_value = embeddings.get(concept["name"])
                if embedding_value is None:
                    continue
                
                # Use a properly typed variable for the embedding
                embedding = embedding_value
                
                # Add to graph
                concept_id = await self.graph.add_concept(
                    concept["content"],
                    embedding,
                    concept["metadata"]
                )
                
                # Store mapping and added node
                concept_id_map[concept["name"]] = concept_id
                node = await self.graph.get_concept(concept_id)
                if node:
                    added_nodes.append(node)
            
            except Exception as e:
                log.error(f"Failed to integrate concept {concept['name']}: {e}")
                continue
        
        # Add relationships
        for concept in concepts:
            if "relationships" in concept and concept["name"] in concept_id_map:
                source_id = concept_id_map[concept["name"]]
                
                for rel in concept["relationships"]:
                    try:
                        # Skip if target doesn't exist
                        if rel["target"] not in concept_id_map:
                            continue
                        
                        target_id = concept_id_map[rel["target"]]
                        
                        # Add relationship
                        await self.graph.add_relationship(
                            source_id,
                            target_id,
                            rel["type"],
                            {"description": rel.get("description", "")}
                        )
                    except Exception as e:
                        log.error(f"Failed to add relationship: {e}")
                        continue
        
        return added_nodes
    
    def _build_generation_prompt(
        self,
        focus_concept: str,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for concept generation.
        
        Args:
            focus_concept: Current focus concept
            state: Current graph state
            context: Optional context
            
        Returns:
            str: Formatted prompt
        """
        # Get related concepts for context
        related_concepts = state.get("related_concepts", [])
        related_concepts_str = ", ".join(related_concepts[:5]) if related_concepts else "None"
        
        # Format bridge nodes
        bridge_nodes = state.get("bridge_nodes", [])
        bridge_nodes_str = ", ".join(bridge_nodes[:3]) if bridge_nodes else "None"
        
        # Use the iterative reasoning prompt
        prompt = ITERATIVE_REASONING_PROMPT.format(
            focus_concept=focus_concept,
            related_concepts=related_concepts_str,
            bridge_nodes=bridge_nodes_str,
            modularity=f"{state['modularity']:.2f}",
            avg_path_length=f"{state['avg_path_length']:.2f}"
        )

        # Add additional context if provided
        if context:
            prompt += "\n\nAdditional context:\n"
            for k, v in context.items():
                prompt += f"- {k}: {v}\n"

        return prompt
    
    async def _perform_compositional_reasoning(
        self,
        node_ids: List[str]
    ) -> Dict[str, Any]:
        """Perform compositional reasoning on significant nodes.
        
        Args:
            node_ids: List of significant node IDs
            
        Returns:
            Dict[str, Any]: Results of compositional reasoning
        """
        if not self.enable_compositional:
            return {"success": False, "error": "Compositional reasoning not enabled"}
        
        try:
            # Sample nodes if too many
            if len(node_ids) > 10:
                # Prioritize bridge nodes and high-centrality nodes
                state = await self.graph.get_graph_state()
                bridge_nodes = state.get("bridge_nodes", [])
                
                # Include bridge nodes first
                selected_nodes = [node_id for node_id in node_ids if node_id in bridge_nodes]
                
                # Add random nodes to reach 10
                remaining = list(set(node_ids) - set(selected_nodes))
                if remaining and len(selected_nodes) < 10:
                    random.shuffle(remaining)
                    selected_nodes.extend(remaining[:10 - len(selected_nodes)])
                
                node_ids = selected_nodes[:10]
            
            # Perform compositional reasoning
            results = await self.compositional.synthesize_concepts(node_ids)
            
            # Process and integrate results if successful
            if results.get("success", False) and "results" in results:
                for result in results["results"]:
                    if "content" in result:
                        # Parse entities and relationships
                        content = result["content"]
                        entities, relationships = self.parser.parse_response(content)
                        
                        # Create concepts dictionary for integration
                        concepts = []
                        for entity in entities:
                            concepts.append({
                                "name": entity["name"],
                                "content": entity["content"],
                                "metadata": {
                                    **entity["metadata"],
                                    "compositional": True,
                                    "source_concepts": result.get("source_concepts", [])
                                },
                                "relationships": [
                                    rel for rel in relationships 
                                    if rel["source"] == entity["name"] or rel["target"] == entity["name"]
                                ]
                            })
                        
                        # Integrate compositional concepts
                        await self._integrate_concepts(concepts)
            
            return results
            
        except Exception as e:
            log.error(f"Compositional reasoning failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _check_stability(self) -> bool:
        """Check if graph has reached stable state.

        Returns:
            bool: True if stable
        """
        if len(self.metric_history) < self.stability_window:
            return False

        # Get recent metrics
        recent = self.metric_history[-self.stability_window:]

        # Check path length stability
        path_lengths = [m["avg_path_length"] for m in recent]
        if not all(
            self.min_path_length <= path_len <= self.max_path_length
            for path_len in path_lengths
        ):
            return False

        # Check diameter stability
        diameters = [m.get("diameter", 0) for m in recent]
        if not all(
            self.min_diameter <= d <= self.max_diameter
            for d in diameters
        ):
            return False

        return True
