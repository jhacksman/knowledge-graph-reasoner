#!/usr/bin/env python3
"""
Fix script for test file constructor errors.

This script fixes missing constructor arguments in test files.
"""
import os
import re
from pathlib import Path


def fix_test_constructors():
    """Fix missing constructor arguments in test files."""
    # Define the paths to the test files
    test_files = [
        Path("tests/test_models.py"),
        Path("tests/test_api/test_models.py")
    ]
    
    for test_file in test_files:
        if not test_file.exists():
            print(f"Error: {test_file} does not exist")
            continue
        
        # Read the current content of the file
        with open(test_file, "r") as file:
            content = file.read()
        
        # Fix missing Node constructor arguments
        content = re.sub(
            r"Node\(id=([^,\)]+)\)",
            r"Node(id=\1, content=\"Test content\")",
            content
        )
        
        # Fix missing Edge constructor arguments
        content = re.sub(
            r"Edge\(source=([^,\)]+)\)",
            r"Edge(source=\1, target=\"target\", type=\"related\")",
            content
        )
        
        content = re.sub(
            r"Edge\(source=([^,\)]+), target=([^,\)]+)\)",
            r"Edge(source=\1, target=\2, type=\"related\")",
            content
        )
        
        # Save the updated content
        with open(test_file, "w") as file:
            file.write(content)
        
        print(f"Fixed constructor errors in {test_file}")


if __name__ == "__main__":
    fix_test_constructors()