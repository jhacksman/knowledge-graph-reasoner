"""Tests for the extraction module."""
import unittest
import numpy as np

from src.extraction.parser import EntityRelationshipParser
from src.extraction.deduplication import DeduplicationHandler
from src.models.node import Node


class TestEntityRelationshipParser(unittest.TestCase):
    """Tests for the EntityRelationshipParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = EntityRelationshipParser()

    def test_parse_entity(self):
        """Test parsing of entity format."""
        response = "<entity>concept1: description1</entity>"
        entities, relationships = self.parser.parse_response(response)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["name"], "concept1")
        self.assertEqual(entities[0]["content"], "description1")
        self.assertEqual(len(relationships), 0)

    def test_parse_relationship(self):
        """Test parsing of relationship format."""
        response = "<relationship>concept1: concept2: relates_to: description</relationship>"
        entities, relationships = self.parser.parse_response(response)

        self.assertEqual(len(entities), 0)
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0]["source"], "concept1")
        self.assertEqual(relationships[0]["target"], "concept2")
        self.assertEqual(relationships[0]["type"], "relates_to")
        self.assertEqual(relationships[0]["description"], "description")

    def test_parse_multiple(self):
        """Test parsing of multiple entities and relationships."""
        response = """
        <entity>concept1: description1</entity>
        <entity>concept2: description2</entity>
        <relationship>concept1: concept2: relates_to:
            description</relationship>
        
        <relationship>concept2: concept1: depends_on: another description</relationship>
        """
        entities, relationships = self.parser.parse_response(response)

        self.assertEqual(len(entities), 2)
        self.assertEqual(len(relationships), 2)

        # Check entities
        self.assertEqual(entities[0]["name"], "concept1")
        self.assertEqual(entities[0]["content"], "description1")
        self.assertEqual(entities[1]["name"], "concept2")
        self.assertEqual(entities[1]["content"], "description2")

        # Check relationships
        self.assertEqual(relationships[0]["source"], "concept1")
        self.assertEqual(relationships[0]["target"], "concept2")
        self.assertEqual(relationships[0]["type"], "relates_to")
        self.assertEqual(relationships[1]["source"], "concept2")
        self.assertEqual(relationships[1]["target"], "concept1")
        self.assertEqual(relationships[1]["type"], "depends_on")

    def test_parse_empty(self):
        """Test parsing of empty response."""
        response = ""
        entities, relationships = self.parser.parse_response(response)

        self.assertEqual(len(entities), 0)
        self.assertEqual(len(relationships), 0)

    def test_parse_malformed(self):
        """Test parsing of malformed response."""
        response = """
        <entity>malformed entity</entity>
        <relationship>malformed relationship</relationship>
        """
        entities, relationships = self.parser.parse_response(response)

        self.assertEqual(len(entities), 0)
        self.assertEqual(len(relationships), 0)


class TestDeduplicationHandler(unittest.TestCase):
    """Tests for the DeduplicationHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = DeduplicationHandler(similarity_threshold=0.85)

    def test_find_duplicates(self):
        """Test duplicate detection with similarity threshold of 0.85."""
        # Create test data
        new_entities = [
            {"name": "concept1", "content": "description1"},
            {"name": "concept2", "content": "description2"}
        ]

        existing_entities = [
            Node(id="1", content="existing1", metadata={"embedding": [0.9, 0.1]}),
            Node(id="2", content="existing2", metadata={"embedding": [0.1, 0.9]})
        ]

        # Create embeddings with high similarity to first existing entity
        embeddings = {
            "concept1": np.array([0.95, 0.05]),  # Similar to existing1
            "concept2": np.array([0.2, 0.8])     # Not similar enough to existing2
        }

        duplicates = self.handler.find_duplicates(
            new_entities, existing_entities, embeddings
        )

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates["concept1"], "1")
        self.assertNotIn("concept2", duplicates)

    def test_find_duplicates_threshold(self):
        """Test duplicate detection with different similarity thresholds."""
        # Create test data
        new_entities = [
            {"name": "concept1", "content": "description1"}
        ]

        existing_entities = [
            Node(id="1", content="existing1", metadata={"embedding": [0.8, 0.2]})
        ]

        embeddings = {
            "concept1": np.array([0.9, 0.1])  # Similarity = 0.98
        }

        # Test with higher threshold (0.99)
        handler_high = DeduplicationHandler(similarity_threshold=0.99)
        duplicates_high = handler_high.find_duplicates(
            new_entities, existing_entities, embeddings
        )
        self.assertEqual(len(duplicates_high), 0)  # Not similar enough

        # Test with lower threshold (0.80)
        handler_low = DeduplicationHandler(similarity_threshold=0.80)
        duplicates_low = handler_low.find_duplicates(
            new_entities, existing_entities, embeddings
        )
        self.assertEqual(len(duplicates_low), 1)  # Similar enough

    def test_missing_embeddings(self):
        """Test handling of missing embeddings."""
        new_entities = [
            {"name": "concept1", "content": "description1"},
            {"name": "concept2", "content": "description2"}
        ]

        existing_entities = [
            Node(id="1", content="existing1", metadata={"embedding": [0.9, 0.1]})
        ]

        # Only provide embedding for concept1
        embeddings = {
            "concept1": np.array([0.95, 0.05])
        }

        duplicates = self.handler.find_duplicates(
            new_entities, existing_entities, embeddings
        )

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates["concept1"], "1")
        # No embedding, so not in duplicates
        self.assertNotIn("concept2", duplicates)

    def test_empty_embeddings(self):
        """Test handling of empty embeddings."""
        new_entities = [
            {"name": "concept1", "content": "description1"}
        ]

        existing_entities = [
            Node(id="1", content="existing1", metadata={})  # No embedding
        ]

        embeddings = {
            "concept1": np.array([0.95, 0.05])
        }

        duplicates = self.handler.find_duplicates(
            new_entities, existing_entities, embeddings
        )

        # No duplicates found due to missing embedding
        self.assertEqual(len(duplicates), 0)


if __name__ == "__main__":
    unittest.main()
