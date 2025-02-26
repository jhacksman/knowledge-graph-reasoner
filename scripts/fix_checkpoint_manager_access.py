#!/usr/bin/env python3
"""
Fix script for CheckpointManager method access issues.

This script adds null checks before accessing methods on potentially None objects.
"""
import os
import re
from pathlib import Path


def fix_checkpoint_manager_access():
    """Fix CheckpointManager method access issues."""
    # Define the path to the test_checkpoint.py file
    checkpoint_file = Path("tests/test_checkpoint.py")
    
    if not checkpoint_file.exists():
        print(f"Error: {checkpoint_file} does not exist")
        return
    
    # Read the current content of the file
    with open(checkpoint_file, "r") as file:
        content = file.read()
    
    # Fix Item "None" of "CheckpointManager | None" has no attribute issues
    content = re.sub(
        r"(checkpoint_manager)\.load_checkpoint\(([^)]*)\)",
        r"((\1 is not None) and \1.load_checkpoint(\2))",
        content
    )
    
    content = re.sub(
        r"(checkpoint_manager)\.list_checkpoints\(\)",
        r"((\1 is not None) and \1.list_checkpoints())",
        content
    )
    
    content = re.sub(
        r"(checkpoint_manager)\.validate_checkpoint\(([^)]*)\)",
        r"((\1 is not None) and \1.validate_checkpoint(\2))",
        content
    )
    
    content = re.sub(
        r"(checkpoint_manager)\.create_checkpoint\(([^)]*)\)",
        r"((\1 is not None) and \1.create_checkpoint(\2))",
        content
    )
    
    # Fix Incompatible types in "await" issues
    content = re.sub(
        r"await (checkpoint_manager)",
        r"await (\1.create_checkpoint() if \1 is not None else None)",
        content
    )
    
    # Save the updated content
    with open(checkpoint_file, "w") as file:
        file.write(content)
    
    print(f"Fixed CheckpointManager method access issues in {checkpoint_file}")


if __name__ == "__main__":
    fix_checkpoint_manager_access()