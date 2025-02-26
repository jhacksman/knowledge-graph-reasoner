#!/usr/bin/env python3
"""
Fix script for MilvusVectorStore vs MilvusStore naming inconsistency.

This script replaces all occurrences of MilvusVectorStore with MilvusStore
in the API route files.
"""
import os
import re
from pathlib import Path


def fix_milvus_store_name():
    """Fix MilvusVectorStore vs MilvusStore naming inconsistency."""
    # Define the pattern to search for
    pattern = r"MilvusVectorStore"
    replacement = "MilvusStore"
    
    # Define the directories to search in
    api_dir = Path("src/api")
    
    # Find all Python files in the API directory
    python_files = list(api_dir.glob("**/*.py"))
    
    # Add main.py if it exists
    if (api_dir / "main.py").exists():
        python_files.append(api_dir / "main.py")
    
    # Process each file
    for file_path in python_files:
        with open(file_path, "r") as file:
            content = file.read()
        
        # Check if the pattern exists in the file
        if re.search(pattern, content):
            # Replace the pattern
            new_content = re.sub(pattern, replacement, content)
            
            # Write the updated content back to the file
            with open(file_path, "w") as file:
                file.write(new_content)
            
            print(f"Fixed MilvusVectorStore in {file_path}")


if __name__ == "__main__":
    fix_milvus_store_name()
