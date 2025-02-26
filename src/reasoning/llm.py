"""Venice.ai LLM integration."""
from typing import Dict, Any, List, Optional
import aiohttp
import numpy as np


class VeniceLLMConfig:
    """Configuration for Venice.ai LLM client."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-r1-671b",
        base_url: str = "https://api.venice.ai/api/v1",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """Initialize LLM config.
        
        Args:
            api_key: Venice.ai API key
            model: Model name to use
            base_url: Base API URL
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout


class VeniceLLM:
    """Venice.ai LLM client."""
    
    def __init__(self, config: VeniceLLMConfig):
        """Initialize LLM client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=timeout
            )
            self._session = session
    
    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Get text embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Text embedding
        """
        await self._ensure_session()
        if not self._session:
            raise RuntimeError("Failed to initialize session")
            
        # Get response using post
        response = await self._session.post(
            f"{self.config.base_url}/embeddings",
            json={
                "model": self.config.model,
                "input": text
            }
        )
        response.raise_for_status()
        data = await response.json()
        return np.array(data["data"][0]["embedding"])
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate text completion.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict[str, Any]: Response data
        """
        await self._ensure_session()
        if not self._session:
            raise RuntimeError("Failed to initialize session")
            
        # Get response using post
        response = await self._session.post(
            f"{self.config.base_url}/chat/completions",
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        data = await response.json()

        return dict(data)
