"""Deduplication handling for knowledge graph entities."""
from typing import List, Dict, Any
import numpy as np
from ..models.node import Node


class DeduplicationHandler:
    """Handles deduplication of entities in the knowledge graph."""

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize deduplication handler.

        Args:
            similarity_threshold: Threshold for considering entities as duplicates
        """
        self.similarity_threshold = similarity_threshold

    def find_duplicates(
        self,
        new_entities: List[Dict[str, Any]],
        existing_entities: List[Node],
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        """Find duplicate entities.

        Args:
            new_entities: New entities to check
            existing_entities: Existing entities in the graph
            embeddings: Embeddings for all entities

        Returns:
            Dict[str, str]: Mapping from new entity names to existing entity IDs
        """
        duplicates = {}

        for new_entity in new_entities:
            new_embedding = embeddings.get(new_entity["name"])
            if new_embedding is None:
                continue

            for existing_entity in existing_entities:
                existing_embedding = np.array(
                    existing_entity.metadata.get(
                    "embedding", [])
                )
                if len(existing_embedding) == 0:
                    continue

                # Compute cosine similarity
                similarity = np.dot(new_embedding, existing_embedding) / (
                    np.linalg.norm(new_embedding) *
                    np.linalg.norm(existing_embedding)
                )

                if similarity >= self.similarity_threshold:
                    duplicates[new_entity["name"]] = existing_entity.id
                    break

        return duplicates
