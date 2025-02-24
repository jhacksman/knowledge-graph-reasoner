"""Query execution pipeline."""
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np

from ..models.node import Node
from ..models.edge import Edge
from ..reasoning.decomposition import QueryDecomposer
from ..reasoning.llm import VeniceLLM
from ..vector_store.base import BaseVectorStore


class QueryExecutor:
    """Executes graph queries using decomposition and vector search."""
    
    def __init__(
        self,
        llm: VeniceLLM,
        vector_store: BaseVectorStore,
        search_threshold: float = 0.5,
        max_results: int = 5
    ):
        """Initialize executor.
        
        Args:
            llm: LLM client for reasoning
            vector_store: Vector store for graph operations
            search_threshold: Similarity threshold for vector search
            max_results: Maximum number of results per subquery
        """
        self.llm = llm
        self.vector_store = vector_store
        self.decomposer = QueryDecomposer(llm)
        self.search_threshold = search_threshold
        self.max_results = max_results
    
    async def execute_query(
        self,
        query: str,
        context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Execute a query against the knowledge graph.
        
        Args:
            query: Query to execute
            context: Optional context for query execution
            
        Returns:
            Dict[str, Any]: Query results with explanations
        """
        # Decompose query into subqueries
        subqueries = await self.decomposer.decompose_query(query, context)
        
        # Execute each subquery
        results = []
        for subquery in subqueries:
            # Get embedding for subquery
            embedding = await self.decomposer.get_query_embedding(subquery["text"])
            
            # Search for relevant nodes
            nodes = await self.vector_store.search_similar(
                embedding,
                k=self.max_results,
                threshold=self.search_threshold
            )
            
            # Get edges for found nodes
            edges = []
            seen_edges = set()  # Track unique edges
            for node in nodes:
                async for edge in self.vector_store.get_edges(source_id=node.id):
                    edge_key = f"{edge.source}_{edge.target}_{edge.type}"
                    if edge_key not in seen_edges:
                        edges.append(edge)
                        seen_edges.add(edge_key)
            
            # Add results
            results.append({
                "subquery": subquery["text"],
                "nodes": nodes,
                "edges": edges
            })
        
        # Generate final response
        messages = [{
            "role": "system",
            "content": (
                "You are a knowledge graph reasoning system. "
                "Your task is to synthesize information from subquery results "
                "into a coherent response that answers the original query."
            )
        }, {
            "role": "user",
            "content": self._format_results_prompt(query, results)
        }]
        
        response = await self.llm.generate(messages)
        
        return {
            "query": query,
            "subqueries": subqueries,
            "results": results,
            "explanation": response["choices"][0]["message"]["content"]
        }
    
    def _format_results_prompt(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Format results into a prompt for final synthesis.
        
        Args:
            query: Original query
            results: Subquery results
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"Original query: {query}\n\nResults from subqueries:\n"
        
        for result in results:
            prompt += f"\nSubquery: {result['subquery']}\n"
            prompt += "Relevant information:\n"
            
            # Add node information
            for node in result["nodes"]:
                prompt += f"- {node.content}\n"
            
            # Add relationship information
            for edge in result["edges"]:
                prompt += f"- Relationship: {edge.type} between {edge.source} and {edge.target}\n"
        
        prompt += "\nPlease synthesize this information to answer the original query."
        return prompt
