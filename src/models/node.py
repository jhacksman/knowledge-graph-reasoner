"""Node model for the knowledge graph."""
from typing import Dict, Any
from pydantic import BaseModel, Field


class Node(BaseModel):
    """A node in the knowledge graph."""
    
    id: str = Field(description="Unique identifier for the node")
    content: str = Field(description="Content/text of the node")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the node"
    )
