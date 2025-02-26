#!/usr/bin/env python3
"""
Fix script for Permission enum issues.

This script adds the missing attributes to the Permission enum in src/api/auth.py.
"""
import os
import re
from pathlib import Path
from enum import Enum, auto


def fix_permission_enum():
    """Fix Permission enum issues."""
    # Define the path to the auth.py file
    auth_file = Path("src/api/auth.py")
    
    if not auth_file.exists():
        print(f"Error: {auth_file} does not exist")
        return
    
    # Read the current content of the file
    with open(auth_file, "r") as file:
        content = file.read()
    
    # Define the pattern to search for
    pattern = r"class Permission\(Enum\):[\s\S]*?(?=\n\n)"
    
    # Define the replacement enum
    replacement = """class Permission(Enum):
    """
    
    # Add all the missing permissions
    permissions = [
        "READ_PERMISSION_CONCEPTS",
        "WRITE_PERMISSION_CONCEPTS",
        "READ_PERMISSION_RELATIONSHIPS",
        "WRITE_PERMISSION_RELATIONSHIPS",
        "READ_PERMISSION_QUERIES",
        "WRITE_PERMISSION_QUERIES",
        "READ_PERMISSION_METRICS",
        "WRITE_PERMISSION",
        "ADMIN_PERMISSION_ACCESS",
    ]
    
    for permission in permissions:
        replacement += f"    {permission} = auto()\n"
    
    # Replace the enum in the content
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        # Write the updated content back to the file
        with open(auth_file, "w") as file:
            file.write(new_content)
        
        print(f"Fixed Permission enum in {auth_file}")
    else:
        print(f"Error: Could not find Permission enum in {auth_file}")


if __name__ == "__main__":
    fix_permission_enum()
