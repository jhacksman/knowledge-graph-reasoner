"""Edge model for the knowledge graph."""
from typing import Dict, Any
from pydantic import BaseModel, Field


class Edge(BaseModel):
    """An edge in the knowledge graph."""
    
    source: str = Field(description="ID of the source node")
    target: str = Field(description="ID of the target node")
    type: str = Field(description="Type of relationship")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the edge"
    )
