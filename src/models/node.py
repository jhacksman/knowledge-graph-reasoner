"""Node representation for the knowledge graph."""
from typing import Dict, Optional
import numpy as np
from pydantic import BaseModel, Field


class Node(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str = Field(description="Unique identifier for the node")
    embedding: np.ndarray = Field(description="Vector representation of the node content")
    content: str = Field(description="Text content of the node")
    metadata: Dict = Field(default_factory=dict, description="Additional node metadata")
    
    class Config:
        arbitrary_types_allowed = True  # Needed for numpy array support
