"""Main reasoning pipeline for knowledge graph expansion."""
from typing import Dict, List, Optional, Tuple
import asyncio
from pydantic import BaseModel

from ..vector_store.milvus_store import MilvusGraphStore
from ..reasoning.llm import VeniceLLM
from ..models.node import Node
from ..models.edge import Edge


class KnowledgeExpansionResult(BaseModel):
    """Result of a knowledge graph expansion operation."""
    new_nodes: List[Node]
    new_edges: List[Edge]
    query_decomposition: List[str]
    reasoning_context: List[str]
    final_reasoning: str


class ReasoningPipeline:
    """Main pipeline for knowledge graph reasoning and expansion."""
    
    def __init__(self, store: MilvusGraphStore, llm: VeniceLLM):
        """Initialize the reasoning pipeline.
        
        Args:
            store: Vector store for the knowledge graph
            llm: LLM client for reasoning
        """
        self.store = store
        self.llm = llm
    
    async def _extract_knowledge(self, reasoning: str) -> Tuple[List[Node], List[Edge]]:
        """Extract structured knowledge from reasoning output.
        
        Args:
            reasoning: Raw reasoning output from LLM
            
        Returns:
            Tuple[List[Node], List[Edge]]: Extracted nodes and edges
        """
        # Use LLM to extract structured knowledge
        prompt = f"""Extract key concepts and relationships from the following text.
        Format the response as a JSON object with two arrays:
        1. "nodes": Array of objects with "content" and "metadata" fields
        2. "edges": Array of objects with "source", "target", "type", and "metadata" fields
        
        Text: {reasoning}
        
        Think step by step:
        1. Identify key concepts (nodes)
        2. Identify relationships between concepts (edges)
        3. Structure the information
        
        Format your response as valid JSON:
        """
        
        extraction = await self.llm.reason_over_context(prompt, [reasoning])
        try:
            structured = json.loads(extraction)
            
            nodes = [
                Node(
                    id=f"node_{i}_{hash(n['content'])}", 
                    content=n["content"],
                    embedding=await self.llm.embed_text(n["content"]),
                    metadata=n.get("metadata", {})
                )
                for i, n in enumerate(structured["nodes"])
            ]
            
            edges = [
                Edge(
                    source=f"node_{i}_{hash(e['source'])}",
                    target=f"node_{i}_{hash(e['target'])}",
                    type=e["type"],
                    metadata=e.get("metadata", {})
                )
                for i, e in enumerate(structured["edges"])
            ]
            
            return nodes, edges
        except Exception as e:
            raise RuntimeError(f"Failed to extract knowledge: {e}")
    
    async def expand_graph(
        self,
        query: str,
        max_context_nodes: int = 5,
        similarity_threshold: float = 0.7
    ) -> KnowledgeExpansionResult:
        """Expand the knowledge graph based on a query.
        
        Args:
            query: Query to expand the graph with
            max_context_nodes: Maximum number of similar nodes to include
            similarity_threshold: Minimum similarity for context nodes
            
        Returns:
            KnowledgeExpansionResult: Results of the expansion
        """
        # 1. Decompose query into sub-queries
        sub_queries = await self.llm.decompose_query(query)
        
        # 2. Search for relevant context nodes
        context_nodes = []
        for sub_query in sub_queries:
            # Get embedding for sub-query
            sub_query_embedding = await self.llm.embed_text(sub_query)
            
            # Search for similar nodes
            similar_nodes = await self.store.search_similar(
                embedding=sub_query_embedding,
                k=max_context_nodes,
                threshold=similarity_threshold
            )
            context_nodes.extend(similar_nodes)
        
        # Remove duplicates while preserving order
        seen = set()
        context_nodes = [
            node for node in context_nodes
            if not (node.id in seen or seen.add(node.id))
        ]
        
        # 3. Generate new knowledge through reasoning
        context_texts = [node.content for node in context_nodes]
        reasoning = await self.llm.reason_over_context(query, context_texts)
        
        # 4. Extract and structure new knowledge
        new_nodes, new_edges = await self._extract_knowledge(reasoning)
        
        # 5. Add new knowledge to graph
        for node in new_nodes:
            await self.store.add_node(node)
        
        for edge in new_edges:
            await self.store.add_edge(edge)
        
        return KnowledgeExpansionResult(
            new_nodes=new_nodes,
            new_edges=new_edges,
            query_decomposition=sub_queries,
            reasoning_context=context_texts,
            final_reasoning=reasoning
        )
    
    async def expand_graph_iteratively(
        self,
        query: str,
        max_iterations: int = 3,
        **kwargs
    ) -> List[KnowledgeExpansionResult]:
        """Expand the graph iteratively, using new knowledge for further expansion.
        
        Args:
            query: Initial query to expand from
            max_iterations: Maximum number of expansion iterations
            **kwargs: Additional arguments for expand_graph
            
        Returns:
            List[KnowledgeExpansionResult]: Results from each iteration
        """
        results = []
        current_query = query
        
        for i in range(max_iterations):
            # Expand graph with current query
            result = await self.expand_graph(current_query, **kwargs)
            results.append(result)
            
            # Generate next query based on new knowledge
            next_query_prompt = f"""Based on the original question and our current knowledge,
            what follow-up question would help us explore important related concepts?
            
            Original question: {query}
            Current knowledge:
            {result.final_reasoning}
            
            Think step by step:
            1. What aspects haven't been fully explored?
            2. What related concepts need more investigation?
            3. What logical next questions arise?
            
            Provide a single, focused follow-up question:
            """
            
            current_query = await self.llm.reason_over_context(
                next_query_prompt,
                [result.final_reasoning]
            )
            
            # Stop if no meaningful expansion is possible
            if not result.new_nodes and not result.new_edges:
                break
        
        return results
