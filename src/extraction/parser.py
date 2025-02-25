"""Entity-relationship extraction from LLM responses."""
from typing import List, Dict, Any, Tuple, Optional, Union
import re
import json
import logging

log = logging.getLogger(__name__)


class EntityRelationshipParser:
    """Extracts entities and relationships from LLM responses."""

    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize parser.
        
        Args:
            confidence_threshold: Minimum confidence score for relationships
        """
        self.confidence_threshold = confidence_threshold
        
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

        # Try structured JSON format first
        json_entities, json_relationships = self._parse_json_format(response)
        if json_entities or json_relationships:
            return json_entities, json_relationships

        # Fall back to XML-like format
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
                "metadata": {"confidence": 1.0}  # Default confidence for backward compatibility
            })

        # Try natural language parsing if no structured format found
        if not entities and not relationships:
            nl_entities, nl_relationships = self._parse_natural_language(response)
            entities.extend(nl_entities)
            relationships.extend(nl_relationships)

        return entities, relationships
        
    def _parse_json_format(
        self, response: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse JSON format in LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Extracted 
            entities and relationships
        """
        entities = []
        relationships = []
        
        # Look for JSON blocks in the response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.finditer(json_pattern, response, re.DOTALL)
        
        for match in json_matches:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)
                
                # Extract entities
                if "entities" in data:
                    for entity in data["entities"]:
                        if "name" in entity and "content" in entity:
                            entities.append({
                                "name": entity["name"],
                                "content": entity["content"],
                                "metadata": entity.get("metadata", {})
                            })
                
                # Extract relationships
                if "relationships" in data:
                    for rel in data["relationships"]:
                        if all(k in rel for k in ["source", "target", "type"]):
                            # Filter by confidence if available
                            confidence = rel.get("confidence", 1.0)
                            if confidence >= self.confidence_threshold:
                                relationships.append({
                                    "source": rel["source"],
                                    "target": rel["target"],
                                    "type": rel["type"],
                                    "description": rel.get("description", ""),
                                    "metadata": {
                                        **rel.get("metadata", {}),
                                        "confidence": confidence
                                    }
                                })
            except json.JSONDecodeError:
                log.warning("Failed to parse JSON in response")
                continue
        
        return entities, relationships
        
    def _parse_natural_language(
        self, response: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and relationships from natural language.
        
        Args:
            response: Natural language text
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Extracted 
            entities and relationships
        """
        # This is a placeholder for more sophisticated NLP-based extraction
        # In a real implementation, this would use NER and relation extraction
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        
        # Simple heuristic extraction for demonstration
        # Look for "X is a Y" patterns
        is_a_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+a\s+([A-Z][a-z]+(?:\s+[a-z]+)*)'
        for match in re.finditer(is_a_pattern, response):
            entity_name = match.group(1).strip()
            entity_type = match.group(2).strip()
            
            # Add entity if not already present
            if not any(e["name"] == entity_name for e in entities):
                entities.append({
                    "name": entity_name,
                    "content": f"{entity_name} is a {entity_type}",
                    "metadata": {"type": entity_type}
                })
        
        # Look for "X relates to Y" patterns
        relates_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(relates|connects|links)\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        for match in re.finditer(relates_pattern, response):
            source = match.group(1).strip()
            relation = match.group(2).strip()
            target = match.group(3).strip()
            
            relationships.append({
                "source": source,
                "target": target,
                "type": relation,
                "description": f"{source} {relation} to {target}",
                "metadata": {"confidence": 0.7}  # Lower confidence for heuristic extraction
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
