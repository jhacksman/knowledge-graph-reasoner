"""Graph query decomposition."""
from typing import List, Optional
import json

from ..models.node import Node
from ..models.edge import Edge
from .llm import VeniceLLM, VeniceLLMConfig


class GraphQueryDecomposer:
    """Decomposes graph queries into sub-queries for comprehensive exploration."""

    def __init__(self, llm: VeniceLLM):
        """Initialize the decomposer.
        
        Args:
            llm: LLM client for query decomposition
        """
        self.llm = llm

    async def decompose_query(
        self,
        query: str,
        context: Optional[List[str]] = None
    ) -> List[str]:
        """Decompose a graph query into sub-queries.
        
        Args:
            query: Original query to decompose
            context: Optional list of context strings
            
        Returns:
            List of sub-queries
        """
        prompt = """To explore this knowledge graph query more comprehensively, break down the original query into up to four sub-queries that will help discover relevant nodes and relationships. Return as list of str.
If this is a very simple query and no decomposition is necessary, then keep only the original query in the list.

Original Query: {query}
Context: {context}

<EXAMPLE>
Example input:
"Explain deep learning architecture"

Example output:
[
    "What are the key components of deep learning architectures?",
    "How do different layers in a neural network connect and interact?",
    "What are the bridge concepts between traditional and deep learning architectures?",
    "Which architectural patterns are most central to modern deep learning?"
]
</EXAMPLE>

Provide your response in list of str format:
"""
        try:
            response = await self.llm.generate(
                messages=[{
                    "role": "user",
                    "content": prompt.format(
                        query=query,
                        context="\n".join(context) if context else ""
                    )
                }]
            )
            
            # Parse response using ast.literal_eval for safety
            import ast
            sub_queries = ast.literal_eval(response.content)
            if not isinstance(sub_queries, list):
                raise ValueError("Response must be a list of strings")
                
            return sub_queries
        except Exception as e:
            print(f"Failed to decompose query: {e}")
            return [query]  # Fall back to original query
