"""Reasoning pipeline implementation."""
from typing import List, Dict, Any, Optional, Set
import logging
import asyncio
import random

from .llm import VeniceLLM
from ..graph.manager import GraphManager
from ..models.node import Node
from ..extraction.parser import EntityRelationshipParser
from ..extraction.deduplication import DeduplicationHandler
from ..graph.hub_formation import HubFormation
from ..graph.community_preservation import CommunityPreservation

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
        enable_hub_formation: bool = True,
        enable_community_preservation: bool = True,
        target_modularity: float = 0.69,
        target_power_law_exponent: float = 3.0
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
            enable_hub_formation: Enable hub formation mechanisms
            enable_community_preservation: Enable community preservation
            target_modularity: Target modularity score (from paper: ~0.69)
            target_power_law_exponent: Target power law exponent (from paper: ~3.0)
        """
        self.llm = llm
        self.graph = graph
        self.max_iterations = max_iterations
        self.stability_window = stability_window
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.target_modularity = target_modularity
        self.target_power_law_exponent = target_power_law_exponent

        # Add extraction components
        self.parser = EntityRelationshipParser()
        self.deduplication = DeduplicationHandler(similarity_threshold)

        # Add self-organization components
        self.enable_hub_formation = enable_hub_formation
        self.enable_community_preservation = enable_community_preservation

        if enable_hub_formation:
            self.hub_formation = HubFormation(graph.metrics)

        if enable_community_preservation:
            self.community_preservation = CommunityPreservation(graph.metrics.graph)

        # Track metrics history
        self.metric_history: List[Dict[str, Any]] = []

    async def expand_knowledge(
        self,
        seed_concept: str,
        context: Optional[Dict[str, Any]] = None,
        self_organization_interval: int = 5
    ) -> Dict[str, Any]:
        """Iteratively expand knowledge graph.

        Args:
            seed_concept: Initial concept
            context: Optional context for reasoning
            self_organization_interval: Interval for applying self-organization mechanisms

        Returns:
            Dict[str, Any]: Final graph state
        """
        try:
            # Initialize with seed concept
            embedding = await self.llm.embed_text(seed_concept)
            await self.graph.add_concept(seed_concept, embedding)

            # Track focus concept
            focus_concept = seed_concept

            # Track previous communities for evolution analysis
            previous_communities = []

            # Iterative expansion
            for i in range(self.max_iterations):
                log.info(f"Starting iteration {i+1}")

                # Get current graph state
                state = await self.graph.get_graph_state()
                self.metric_history.append(state)

                # Apply self-organization mechanisms at intervals
                if (i > 0 and i % self_organization_interval == 0) or i == self.max_iterations - 1:
                    log.info(f"Applying self-organization at iteration {i + 1}")
                    await self._apply_self_organization(previous_communities)

                    # Update previous communities if available
                    if (self.enable_community_preservation and
                            self.community_preservation.communities):
                        previous_communities = self.community_preservation.communities.copy()

                # Generate new concepts
                concepts = await self._generate_concepts(
                    focus_concept,
                    state,
                    context
                )

                # Add new concepts and relationships
                new_nodes = await self._integrate_concepts(concepts)

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
                        else:
                            # Focus on a new node
                            focus_concept = new_nodes[0].content
                    else:
                        # Focus on a new node
                        focus_concept = new_nodes[0].content

                # Check stability
                if await self._check_stability():
                    log.info("Graph reached stable state")
                    break

                # Rate limiting
                await asyncio.sleep(1)

            # Apply final self-organization
            log.info("Applying final self-organization")
            await self._apply_self_organization(previous_communities)

            # Get final state
            final_state = await self.graph.get_graph_state()

            # Add self-organization metrics to final state
            if self.enable_hub_formation:
                hub_analysis = await self.hub_formation.analyze_hub_structure()
                final_state["hub_analysis"] = hub_analysis

            if self.enable_community_preservation:
                community_metrics = await self.community_preservation.get_community_metrics()
                final_state["community_metrics"] = community_metrics

            return final_state

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
        seed_concept: str,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for concept generation.

        Args:
            seed_concept: Seed concept
            state: Current graph state
            context: Optional context

        Returns:
            str: Formatted prompt
        """
        prompt = f"""Given the current knowledge graph state:
- Modularity: {state['modularity']:.2f}
- Average Path Length: {state['avg_path_length']:.2f}
- Bridge Nodes: {len(state['bridge_nodes'])}

Generate new concepts related to: {seed_concept}

Focus on concepts that would:
1. Strengthen existing knowledge clusters
2. Create meaningful bridges between domains
3. Maintain network stability"""

        if context:
            prompt += "\n\nAdditional context:\n"
            for k, v in context.items():
                prompt += f"- {k}: {v}\n"

        prompt += "\nFormat: Return a list of concepts, one per line."
        return prompt

    async def _apply_self_organization(
        self,
        previous_communities: Optional[List[Set[str]]] = None
    ) -> Dict[str, Any]:
        """Apply self-organization mechanisms to the graph.

        Args:
            previous_communities: Communities from previous iteration for evolution analysis

        Returns:
            Dict[str, Any]: Results of self-organization
        """
        results: Dict[str, Any] = {
            "hub_formation": {},
            "community_preservation": {},
            "modularity_optimization": {},
            "community_evolution": {}
        }

        try:
            # Apply hub formation if enabled
            if self.enable_hub_formation:
                # Identify potential hubs
                hub_ids = await self.hub_formation.identify_potential_hubs()

                # Strengthen hub connections
                hub_results = await self.hub_formation.strengthen_hub_connections(
                    hub_ids,
                    self.graph
                )

                results["hub_formation"] = hub_results

            # Apply community preservation if enabled
            if self.enable_community_preservation:
                # Detect communities
                await self.community_preservation.detect_communities()

                # Preserve community structure
                community_results = await self.community_preservation.preserve_community_structure(
                    self.graph
                )

                results["community_preservation"] = community_results

                # Optimize modularity to target value from paper (~0.69)
                modularity_results = await self.community_preservation.optimize_modularity(
                    self.graph,
                    self.target_modularity
                )

                results["modularity_optimization"] = modularity_results

                # Analyze community evolution if previous communities provided
                if previous_communities:
                    evolution_results = await self.community_preservation.analyze_community_evolution(
                        previous_communities
                    )

                    results["community_evolution"] = evolution_results

            return results

        except Exception as e:
            log.error(f"Failed to apply self-organization: {e}")
            return {
                "error": str(e),
                "hub_formation": None,
                "community_preservation": None
            }

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
