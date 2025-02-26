#!/usr/bin/env python3
"""
Fix script for method signature mismatches.

This script updates method calls in the API route files to match the method signatures
in the GraphManager class.
"""
import os
import re
from pathlib import Path


def fix_method_signatures():
    """Fix method signature mismatches."""
    # Define the method signature mappings
    signature_mappings = [
        # get_similar_concepts
        (
            r"get_similar_concepts\s*\(\s*query_embedding\s*=\s*([^,\)]+)(?:,\s*similarity_threshold\s*=\s*([^,\)]+))?\s*\)",
            lambda match: f"get_similar_concepts({match.group(1)}, threshold={match.group(2) if match.group(2) else '0.7'})"
        ),
        # add_concept
        (
            r"add_concept\s*\(\s*(?:id\s*=\s*([^,\)]+),)?\s*(?:name\s*=\s*([^,\)]+),)?\s*(?:description\s*=\s*([^,\)]+),)?\s*(?:domain\s*=\s*([^,\)]+),)?\s*(?:attributes\s*=\s*([^,\)]+))?\s*\)",
            lambda match: f"add_concept(Node(id={match.group(1) if match.group(1) else 'None'}, name={match.group(2) if match.group(2) else 'None'}, description={match.group(3) if match.group(3) else 'None'}, domain={match.group(4) if match.group(4) else 'None'}, attributes={match.group(5) if match.group(5) else '{}'}))"
        ),
        # add_relationship
        (
            r"add_relationship\s*\(\s*(?:source_id\s*=\s*([^,\)]+),)?\s*(?:target_id\s*=\s*([^,\)]+),)?\s*(?:type\s*=\s*([^,\)]+),)?\s*(?:weight\s*=\s*([^,\)]+),)?\s*(?:attributes\s*=\s*([^,\)]+))?\s*\)",
            lambda match: f"add_relationship(Edge(source_id={match.group(1) if match.group(1) else 'None'}, target_id={match.group(2) if match.group(2) else 'None'}, type={match.group(3) if match.group(3) else 'None'}, weight={match.group(4) if match.group(4) else '1.0'}, attributes={match.group(5) if match.group(5) else '{}'}))"
        ),
        # get_relationships
        (
            r"get_relationships\s*\(\s*(?:type\s*=\s*([^,\)]+),)?\s*(?:skip\s*=\s*([^,\)]+),)?\s*(?:limit\s*=\s*([^,\)]+),)?\s*(?:sort_by\s*=\s*([^,\)]+),)?\s*(?:sort_order\s*=\s*([^,\)]+))?\s*\)",
            lambda match: f"get_relationships()"
        ),
    ]
    
    # Define the directories to search in
    api_dir = Path("src/api")
    
    # Find all Python files in the API directory
    python_files = list(api_dir.glob("**/*.py"))
    
    # Process each file
    for file_path in python_files:
        with open(file_path, "r") as file:
            content = file.read()
        
        # Apply each method signature mapping
        modified = False
        for pattern, replacement_func in signature_mappings:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement_func, content)
                modified = True
        
        # Write the updated content back to the file if modified
        if modified:
            with open(file_path, "w") as file:
                file.write(content)
            
            print(f"Fixed method signatures in {file_path}")


if __name__ == "__main__":
    fix_method_signatures()
