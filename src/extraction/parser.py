"""Entity-relationship extraction from LLM responses."""
from typing import List, Dict, Any, Tuple
import re


class EntityRelationshipParser:
    """Extracts entities and relationships from LLM responses."""

    def parse_response(
        self, response: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse LLM response to extract entities and relationships.

        Args:
            response: LLM response text

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Extracted 
            entities and relationships
        """
        # Extract entities and relationships using regex or structured parsing
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []

        # Look for structured sections in the response
        # Example format: <entity>name: description</entity>
        entity_pattern = r'<entity>(.*?):(.*?)</entity>'
        for match in re.finditer(entity_pattern, response, re.DOTALL):
            name = match.group(1).strip()
            description = match.group(2).strip()
            entities.append({
                "name": name,
                "content": description,
                "metadata": {}
            })

        # Example format: <relationship>source: target: type: description</relationship>
        rel_pattern = r'<relationship>(.*?):(.*?):(.*?):(.*?)</relationship>'
        for match in re.finditer(rel_pattern, response, re.DOTALL):
            source = match.group(1).strip()
            target = match.group(2).strip()
            rel_type = match.group(3).strip()
            description = match.group(4).strip()
            relationships.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "description": description,
                "metadata": {}
            })

        return entities, relationships

    async def extract_entities_and_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities and relationships from text.
        
        Args:
            text: Text to extract entities and relationships from
            
        Returns:
            List[Dict[str, Any]]: Extracted entities with their relationships
        """
        entities, relationships = self.parse_response(text)
        
        # Process entities and add relationships
        result = []
        for entity in entities:
            entity_with_rels = entity.copy()
            entity_with_rels["relationships"] = [
                rel for rel in relationships 
                if rel["source"] == entity["name"] or rel["target"] == entity["name"]
            ]
            result.append(entity_with_rels)
        
        return result

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List[Dict[str, Any]]: Extracted entities
        """
        return await self.extract_entities_and_relationships(text)
