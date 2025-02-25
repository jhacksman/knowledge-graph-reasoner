"""Validation of extracted entities and relationships."""
from typing import List, Dict, Any, Tuple, Optional
import logging

log = logging.getLogger(__name__)


class EntityRelationshipValidator:
    """Validates extracted entities and relationships."""
    
    def __init__(self, min_entity_length: int = 2, max_entity_length: int = 100):
        """Initialize validator.
        
        Args:
            min_entity_length: Minimum length for entity names
            max_entity_length: Maximum length for entity names
        """
        self.min_entity_length = min_entity_length
        self.max_entity_length = max_entity_length
    
    def validate_entity(self, entity: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate an entity.
        
        Args:
            entity: Entity to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check required fields
        if "name" not in entity:
            return False, "Entity missing 'name' field"
            
        if "content" not in entity:
            return False, "Entity missing 'content' field"
        
        # Validate entity name
        name = entity["name"]
        if not isinstance(name, str):
            return False, "Entity name must be a string"
            
        if len(name) < self.min_entity_length:
            return False, f"Entity name too short (min {self.min_entity_length} chars)"
            
        if len(name) > self.max_entity_length:
            return False, f"Entity name too long (max {self.max_entity_length} chars)"
        
        # Validate content
        content = entity["content"]
        if not isinstance(content, str):
            return False, "Entity content must be a string"
            
        if not content:
            return False, "Entity content cannot be empty"
        
        return True, None
    
    def validate_relationship(self, relationship: Dict[str, Any], entities: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Validate a relationship.
        
        Args:
            relationship: Relationship to validate
            entities: List of entities to check against
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check required fields
        required_fields = ["source", "target", "type"]
        for field in required_fields:
            if field not in relationship:
                return False, f"Relationship missing '{field}' field"
        
        # Validate source and target exist in entities
        source = relationship["source"]
        target = relationship["target"]
        
        entity_names = [entity["name"] for entity in entities]
        
        if source not in entity_names:
            return False, f"Source entity '{source}' not found"
            
        if target not in entity_names:
            return False, f"Target entity '{target}' not found"
        
        # Validate relationship type
        rel_type = relationship["type"]
        if not isinstance(rel_type, str):
            return False, "Relationship type must be a string"
            
        if not rel_type:
            return False, "Relationship type cannot be empty"
        
        # Validate description if present
        if "description" in relationship and relationship["description"]:
            description = relationship["description"]
            if not isinstance(description, str):
                return False, "Relationship description must be a string"
        
        # Validate confidence if present
        if "metadata" in relationship and "confidence" in relationship["metadata"]:
            confidence = relationship["metadata"]["confidence"]
            if not isinstance(confidence, (int, float)):
                return False, "Confidence score must be a number"
                
            if confidence < 0 or confidence > 1:
                return False, "Confidence score must be between 0 and 1"
        
        return True, None
    
    def validate_extraction(
        self, 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate extracted entities and relationships.
        
        Args:
            entities: List of entities to validate
            relationships: List of relationships to validate
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]: 
                (valid_entities, valid_relationships, validation_errors)
        """
        valid_entities = []
        valid_relationships = []
        validation_errors = []
        
        # Validate entities
        for i, entity in enumerate(entities):
            is_valid, error = self.validate_entity(entity)
            if is_valid:
                valid_entities.append(entity)
            else:
                validation_errors.append({
                    "type": "entity",
                    "index": i,
                    "error": error or "Unknown validation error"
                })
        
        # Validate relationships
        for i, relationship in enumerate(relationships):
            is_valid, error = self.validate_relationship(relationship, valid_entities)
            if is_valid:
                valid_relationships.append(relationship)
            else:
                validation_errors.append({
                    "type": "relationship",
                    "index": i,
                    "error": error or "Unknown validation error"
                })
        
        return valid_entities, valid_relationships, validation_errors
