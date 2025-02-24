"""Main reasoning pipeline for knowledge graph expansion."""
from typing import Dict, List, Optional, Tuple
import asyncio
import json
import networkx as nx
from pydantic import BaseModel
from community import community_louvain

from ..vector_store.milvus_store import MilvusGraphStore
from ..reasoning.llm import VeniceLLM
from ..models.node import Node
from ..models.edge import Edge
from ..metrics.graph_metrics import GraphMetricsComputer


class KnowledgeExpansionResult(BaseModel):
    """Result of a knowledge graph expansion operation."""
    new_nodes: List[Node]
    new_edges: List[Edge]
    query_decomposition: List[str]
    reasoning_context: List[str]
    final_reasoning: str
    metrics: Dict[str, float]


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
        
        try:
            # Try to parse input as JSON first
            try:
                structured = json.loads(reasoning)
                if isinstance(structured, dict) and "nodes" in structured and "edges" in structured:
                    # Input is already structured, use it directly
                    pass
                else:
                    raise ValueError("Invalid JSON format")
            except json.JSONDecodeError:
                # Input is raw text, extract knowledge using LLM
                extraction = await self.llm.reason_over_context(prompt, [reasoning])
                structured = json.loads(extraction)
                if not isinstance(structured, dict) or "nodes" not in structured or "edges" not in structured:
                    raise ValueError("Invalid extraction format")
            
            # Create nodes with embeddings
            nodes = []
            for i, node_data in enumerate(structured.get("nodes", [])):
                if not isinstance(node_data, dict) or "content" not in node_data:
                    continue
                nodes.append(Node(
                    id=f"node_{i}_{hash(node_data['content'])}",
                    content=node_data["content"],
                    embedding=await self.llm.embed_text(node_data["content"]),
                    metadata=node_data.get("metadata", {})
                ))
            
            # Create edges
            edges = []
            for i, edge_data in enumerate(structured.get("edges", [])):
                if not isinstance(edge_data, dict) or not all(k in edge_data for k in ["source", "target", "type"]):
                    continue
                edges.append(Edge(
                    source=f"node_{i}_{hash(edge_data['source'])}",
                    target=f"node_{i}_{hash(edge_data['target'])}",
                    type=edge_data["type"],
                    metadata=edge_data.get("metadata", {})
                ))
            
            return nodes, edges
        except Exception as e:
            raise RuntimeError(f"Failed to extract knowledge: {e}")
    
    async def expand_graph(
        self,
        query: str,
        max_context_nodes: int = 5,
        similarity_threshold: float = 0.7,
        sub_queries: Optional[List[str]] = None
    ) -> KnowledgeExpansionResult:
        """Expand the knowledge graph based on a query.
        
        Args:
            query: Query to expand the graph with
            max_context_nodes: Maximum number of similar nodes to include
            similarity_threshold: Minimum similarity for context nodes
            sub_queries: Optional pre-decomposed queries (will generate if None)
            
        Returns:
            KnowledgeExpansionResult: Results of the expansion
        """
        # 1. Use provided sub-queries or decompose query
        if sub_queries is None:
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
        
        # Compute metrics for the expanded graph
        metrics = GraphMetricsComputer.compute_metrics(new_nodes, new_edges)
        
        return KnowledgeExpansionResult(
            new_nodes=new_nodes,
            new_edges=new_edges,
            query_decomposition=sub_queries,
            reasoning_context=context_texts,
            final_reasoning=reasoning,
            metrics=metrics
        )
    
    async def expand_graph_iteratively(
        self,
        query: str,
        max_iterations: int = 3,
        **kwargs
    ) -> List[KnowledgeExpansionResult]:
        """Expand the graph iteratively, using new knowledge for further expansion."""
        return await self.expand_graph_parallel(query, max_iterations, **kwargs)
    
    async def expand_graph_parallel(
        self,
        query: str,
        max_iterations: int = 3,
        max_context_nodes: int = 5,
        similarity_threshold: float = 0.7,
        batch_size: int = 256
    ) -> List[KnowledgeExpansionResult]:
        """Expand graph using parallel query processing."""
        all_results = []
        
        # Generate initial sub-queries once
        sub_queries = await self.llm.decompose_query(query)
        if not sub_queries:
            return []
            
        # Process initial queries in parallel
        expansion_tasks = [
            self.expand_graph(
                sq,
                max_context_nodes=max_context_nodes,
                similarity_threshold=similarity_threshold,
                sub_queries=[sq]  # Pass as pre-decomposed query
            )
            for sq in sub_queries
        ]
        expansion_results = await asyncio.gather(*expansion_tasks)
        all_results.extend(expansion_results)
        
        # Stop if no meaningful expansion
        if not any(result.new_nodes or result.new_edges for result in expansion_results):
            return all_results
            
        # Process remaining iterations if needed
        if max_iterations > 1:
            # Generate gap queries based on current results
            gap_prompt = f"""Based on the original query and current knowledge, identify specific questions that would help fill important knowledge gaps.

Original Query: {query}

Current Knowledge:
{chr(10).join(result.final_reasoning for result in expansion_results)}

Return as a list of questions in this format:
[
    "What is the relationship between X and Y?",
    "How does concept A influence concept B?"
]
"""
            try:
                response = await self.llm.reason_over_context(gap_prompt, [])
                import ast
                gap_queries = ast.literal_eval(response)
                if isinstance(gap_queries, list) and gap_queries:
                    # Process gap queries in parallel
                    gap_tasks = [
                        self.expand_graph(
                            gq,
                            max_context_nodes=max_context_nodes,
                            similarity_threshold=similarity_threshold,
                            sub_queries=[gq]  # Pass as pre-decomposed query
                        )
                        for gq in gap_queries[:2]  # Limit to 2 gap queries
                    ]
                    gap_results = await asyncio.gather(*gap_tasks)
                    all_results.extend(gap_results)
            except Exception as e:
                print(f"Failed to process gap queries: {e}")
        
        return all_results       
        return all_results
