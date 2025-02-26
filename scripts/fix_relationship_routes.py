#!/usr/bin/env python3
"""
Fix script for relationship route type errors.

This script adds null checks before passing potentially None values to GraphManager methods.
"""
import os
import re
from pathlib import Path


def fix_relationship_routes():
    """Fix relationship route type errors."""
    # Define the path to the relationships.py file
    relationships_file = Path("src/api/routes/relationships.py")
    
    if not relationships_file.exists():
        print(f"Error: {relationships_file} does not exist")
        return
    
    # Read the current content of the file
    with open(relationships_file, "r") as file:
        content = file.read()
    
    # Fix line 162 and 347: Prevent passing None values to add_relationship
    pattern = r"(\s+)await graph_manager.add_relationship\(source_id, target_id, relationship_type,"
    replacement = r"\1# Check for None values to satisfy type checker\n\1if source_id is None or target_id is None or relationship_type is None:\n\1    raise HTTPException(status_code=400, detail=\"source_id, target_id, and relationship_type are required\")\n\1await graph_manager.add_relationship(source_id, target_id, relationship_type,"
    
    updated_content = re.sub(pattern, replacement, content)
    
    # Save the updated content
    with open(relationships_file, "w") as file:
        file.write(updated_content)
    
    print(f"Fixed relationship route type errors in {relationships_file}")


if __name__ == "__main__":
    fix_relationship_routes()