"""Query decomposition system for knowledge graph reasoning."""
from typing import List, Dict, Any
import numpy as np

from .llm import VeniceLLM


class QueryDecomposer:
    """Decomposes complex queries into subqueries for graph reasoning."""
    
    def __init__(self, llm: VeniceLLM):
        """Initialize decomposer.
        
        Args:
            llm: LLM client for reasoning
        """
        self.llm = llm
    
    async def decompose_query(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Decompose a query into subqueries.
        
        Args:
            query: Query to decompose
            context: Optional context for decomposition
            
        Returns:
            List[Dict[str, Any]]: List of subqueries with metadata
        """
        # Construct prompt for decomposition
        messages = [{
            "role": "system",
            "content": (
                "You are a knowledge graph reasoning system. "
                "Your task is to decompose complex queries into simpler subqueries "
                "that can be answered by searching the knowledge graph. "
                "Each subquery should focus on a specific aspect or relationship."
            )
        }, {
            "role": "user",
            "content": f"Decompose this query into subqueries: {query}"
        }]
        
        if context:
            messages[0]["content"] += (
                "\nContext for decomposition:\n" +
                "\n".join(f"- {k}: {v}" for k, v in context.items())
            )
        
        # Get decomposition from LLM
        response = await self.llm.generate(messages)
        
        # Parse response into subqueries
        subqueries = []
        try:
            content = response["choices"][0]["message"]["content"]
            # Simple parsing - in practice we'd want more structured output
            for line in content.split("\n"):
                if line.strip():
                    subqueries.append({
                        "text": line.strip(),
                        "type": "subquery",
                        "metadata": {}
                    })
        except (KeyError, IndexError) as e:
            print(f"Failed to parse LLM response: {e}")
            return []
        
        return subqueries
    
    async def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query.
        
        Args:
            query: Query to embed
            
        Returns:
            np.ndarray: Query embedding
        """
        return await self.llm.embed_text(query)
