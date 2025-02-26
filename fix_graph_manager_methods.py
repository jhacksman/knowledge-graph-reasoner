#!/usr/bin/env python3
"""
Fix script for GraphManager method name mismatches.

This script replaces method calls in the API route files with the correct method names.
"""
import os
import re
from pathlib import Path


def fix_graph_manager_methods():
    """Fix GraphManager method name mismatches."""
    # Define the method name mappings
    method_mappings = {
        r"get_concepts\s*\(": "get_concept(",
        r"add_node\s*\(": "add_concept(",
        r"search_concepts_by_embedding\s*\(": "get_similar_concepts(",
        r"update_concept\s*\(": "update_node(",
        r"delete_concept\s*\(": "remove_node(",
        r"get_relationship\s*\(": "get_edge(",
        r"update_relationship\s*\(": "update_edge(",
        r"delete_relationship\s*\(": "remove_edge(",
    }
    
    # Define the directories to search in
    api_dir = Path("src/api")
    
    # Find all Python files in the API directory
    python_files = list(api_dir.glob("**/*.py"))
    
    # Process each file
    for file_path in python_files:
        with open(file_path, "r") as file:
            content = file.read()
        
        # Apply each method name mapping
        modified = False
        for pattern, replacement in method_mappings.items():
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
        
        # Write the updated content back to the file if modified
        if modified:
            with open(file_path, "w") as file:
                file.write(content)
            
            print(f"Fixed GraphManager method names in {file_path}")


if __name__ == "__main__":
    fix_graph_manager_methods()
