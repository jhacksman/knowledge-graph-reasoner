"""Reasoning pipeline implementation."""
from typing import List, Dict, Any, Optional
import logging
import asyncio

from .llm import VeniceLLM
from ..graph.manager import GraphManager
from ..models.node import Node

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
        max_diameter: float = 18.0
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
        """
        self.llm = llm
        self.graph = graph
        self.max_iterations = max_iterations
        self.stability_window = stability_window
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        
        # Track metrics history
        self.metric_history: List[Dict[str, Any]] = []
    
    async def expand_knowledge(
        self,
        seed_concept: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Iteratively expand knowledge graph.
        
        Args:
            seed_concept: Initial concept
            context: Optional context for reasoning
            
        Returns:
            Dict[str, Any]: Final graph state
        """
        try:
            # Initialize with seed concept
            embedding = await self.llm.embed_text(seed_concept)
            await self.graph.add_concept(seed_concept, embedding)
            
            # Iterative expansion
            for i in range(self.max_iterations):
                log.info(f"Starting iteration {i+1}")
                
                # Get current graph state
                state = await self.graph.get_graph_state()
                self.metric_history.append(state)
                
                # Generate new concepts
                concepts = await self._generate_concepts(
                    seed_concept,
                    state,
                    context
                )
                
                # Add new concepts and relationships
                await self._integrate_concepts(concepts)
                
                # Check stability
                if await self._check_stability():
                    log.info("Graph reached stable state")
                    break
                
                # Rate limiting
                await asyncio.sleep(1)
            
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
        # Construct prompt
        prompt = self._build_generation_prompt(
            seed_concept,
            state,
            context
        )
        
        # Get LLM response
        response = await self.llm.generate([{
            "role": "system",
            "content": "You are a knowledge graph reasoning system."
        }, {
            "role": "user",
            "content": prompt
        }])
        
        # Parse response
        try:
            content = response["choices"][0]["message"]["content"]
            concepts = []
            
            for line in content.split("\n"):
                if line.strip():
                    concepts.append({
                        "content": line.strip(),
                        "metadata": {}
                    })
            
            return concepts
        except Exception as e:
            log.error(f"Failed to parse LLM response: {e}")
            return []
    
    async def _integrate_concepts(
        self,
        concepts: List[Dict[str, Any]]
    ) -> None:
        """Integrate new concepts into graph.
        
        Args:
            concepts: New concepts to integrate
        """
        for concept in concepts:
            try:
                # Get embedding
                embedding = await self.llm.embed_text(concept["content"])
                
                # Add to graph
                concept_id = await self.graph.add_concept(
                    concept["content"],
                    embedding,
                    concept["metadata"]
                )
                
                # Find similar concepts
                similar = await self.graph.get_similar_concepts(
                    embedding,
                    k=3,
                    threshold=0.7
                )
                
                # Add relationships
                for node in similar:
                    if node.id != concept_id:
                        await self.graph.add_relationship(
                            concept_id,
                            node.id,
                            "related"
                        )
            
            except Exception as e:
                log.error(f"Failed to integrate concept: {e}")
                continue
    
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
        if not all(self.min_path_length <= l <= self.max_path_length
                  for l in path_lengths):
            return False
        
        # Check diameter stability
        diameters = [m.get("diameter", 0) for m in recent]
        if not all(self.min_diameter <= d <= self.max_diameter
                  for d in diameters):
            return False
        
        return True
