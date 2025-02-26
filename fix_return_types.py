#!/usr/bin/env python3
"""
Fix script for return type mismatches.

This script updates return type annotations in the API route files to match the actual return types.
"""
import os
import re
from pathlib import Path


def fix_return_types():
    """Fix return type mismatches."""
    # Define the return type mappings
    return_type_mappings = [
        # Concept -> Node
        (
            r"def\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*->\s*Concept\s*:",
            r"def \1(*args, **kwargs) -> Node:"
        ),
        # list[Concept] -> list[Node]
        (
            r"def\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*->\s*list\[Concept\]\s*:",
            r"def \1(*args, **kwargs) -> list[Node]:"
        ),
        # Relationship -> Edge
        (
            r"def\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*->\s*Relationship\s*:",
            r"def \1(*args, **kwargs) -> Edge:"
        ),
        # list[Relationship] -> list[Edge]
        (
            r"def\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*->\s*list\[Relationship\]\s*:",
            r"def \1(*args, **kwargs) -> list[Edge]:"
        ),
        # Query -> Any
        (
            r"def\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*->\s*Query\s*:",
            r"def \1(*args, **kwargs) -> Any:"
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
        
        # Check if we need to add Any import
        if "from typing import Any" not in content and any("-> Any:" in mapping[1] for mapping in return_type_mappings):
            content = "from typing import Any\n" + content
        
        # Apply each return type mapping
        modified = False
        for pattern, replacement in return_type_mappings:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
        
        # Write the updated content back to the file if modified
        if modified:
            with open(file_path, "w") as file:
                file.write(content)
            
            print(f"Fixed return types in {file_path}")


if __name__ == "__main__":
    fix_return_types()
