"""Edge representation for the knowledge graph."""
from typing import Dict, Optional
from pydantic import BaseModel, Field


class Edge(BaseModel):
    """Represents an edge in the knowledge graph."""
    source: str = Field(description="ID of the source node")
    target: str = Field(description="ID of the target node")
    type: str = Field(description="Type of relationship")
    metadata: Dict = Field(default_factory=dict, description="Additional edge metadata")
