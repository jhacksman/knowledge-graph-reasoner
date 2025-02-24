"""Venice.ai LLM integration for knowledge graph reasoning."""
from typing import List, Dict, Any, Optional
import os
import json
import asyncio
import aiohttp
from pydantic import BaseModel


class VeniceLLMConfig(BaseModel):
    """Configuration for Venice.ai LLM."""
    api_key: str
    model: str = "deepseek-r1-671b"
    base_url: str = "https://api.venice.ai/api/v1"
    max_retries: int = 3
    timeout: int = 30


class VeniceLLM:
    """Client for Venice.ai's LLM API."""
    
    def __init__(self, config: VeniceLLMConfig):
        """Initialize the Venice.ai LLM client.
        
        Args:
            config: Configuration for the LLM client
        """
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        """Create aiohttp session for async context manager."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session for async context manager."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        retries: int = 0
    ) -> Dict[str, Any]:
        """Make a request to the Venice.ai API with retries.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            retries: Current retry count
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            RuntimeError: If request fails after max retries
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
            
        try:
            async with self.session.post(
                f"{self.config.base_url}/{endpoint}",
                json=payload,
                timeout=self.config.timeout
            ) as response:
                if response.status == 429:  # Rate limit
                    if retries < self.config.max_retries:
                        await asyncio.sleep(2 ** retries)  # Exponential backoff
                        return await self._make_request(endpoint, payload, retries + 1)
                    raise RuntimeError("Rate limit exceeded after retries")
                    
                response.raise_for_status()
                return await response.json()
                
        except asyncio.TimeoutError:
            if retries < self.config.max_retries:
                return await self._make_request(endpoint, payload, retries + 1)
            raise RuntimeError("Request timed out after retries")
            
        except Exception as e:
            if retries < self.config.max_retries:
                return await self._make_request(endpoint, payload, retries + 1)
            raise RuntimeError(f"API request failed: {e}")
    
    async def decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries.
        
        Args:
            query: Original query to decompose
            
        Returns:
            List[str]: List of sub-queries
        """
        prompt = f"""To answer this question more comprehensively, break it down into up to four sub-questions.
        Return the sub-questions as a JSON array of strings.
        If this is a simple question that doesn't need decomposition, return an array with just the original question.
        
        Question: {query}
        
        Think step by step:
        1. Is this a complex question that needs decomposition?
        2. What are the key aspects that need to be addressed?
        3. How can we break this down into logical sub-questions?
        
        Format your response as a JSON array of strings.
        """
        
        response = await self._make_request(
            "completions",
            {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI that decomposes complex questions into simpler sub-questions."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to parse LLM response: {e}")
    
    async def reason_over_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None
    ) -> str:
        """Reason over provided context to answer a query.
        
        Args:
            query: Query to answer
            context: List of relevant context strings
            max_tokens: Optional maximum tokens for response
            
        Returns:
            str: Reasoned response
        """
        context_str = "\n\n".join(f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context))
        
        prompt = f"""Based on the following context, answer the question thoughtfully.
        If the context doesn't contain enough information, acknowledge what's missing.
        
        {context_str}
        
        Question: {query}
        
        Think step by step:
        1. What relevant information do we have in the context?
        2. How does this information help answer the question?
        3. What logical conclusions can we draw?
        4. Are there any important caveats or limitations?
        
        Provide a comprehensive answer:
        """
        
        response = await self._make_request(
            "completions",
            {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI that reasons carefully over provided context to answer questions accurately."},
                    {"role": "user", "content": prompt}
                ],
                **({"max_tokens": max_tokens} if max_tokens else {})
            }
        )
        
        try:
            return response["choices"][0]["message"]["content"]
        except KeyError as e:
            raise RuntimeError(f"Failed to parse LLM response: {e}")
