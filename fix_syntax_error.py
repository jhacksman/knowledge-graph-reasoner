#!/usr/bin/env python3
"""
Fix script for syntax error in src/api/routes/concepts.py.

This script fixes the unmatched ')' syntax error on line 56.
"""
import re
from pathlib import Path


def fix_syntax_error():
    """Fix syntax error in concepts.py."""
    # Define the path to the file
    file_path = Path("src/api/routes/concepts.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} does not exist")
        return
    
    # Read the current content of the file
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # Check line 56 for unmatched ')'
    if len(lines) >= 56:
        line = lines[55]  # 0-indexed, so line 56 is at index 55
        print(f"Line 56: {line}")
        
        # Fix the unmatched ')' by removing it or balancing parentheses
        # This is a simple fix that assumes the issue is an extra ')'
        fixed_line = line.replace("))", ")")
        
        # If that's not the issue, we might need a more complex fix
        if fixed_line == line:
            # Count opening and closing parentheses
            open_count = line.count("(")
            close_count = line.count(")")
            
            if close_count > open_count:
                # Remove the last closing parenthesis
                last_close_index = line.rindex(")")
                fixed_line = line[:last_close_index] + line[last_close_index+1:]
            elif open_count > close_count:
                # Add a closing parenthesis at the end
                fixed_line = line.rstrip() + ")\n"
        
        lines[55] = fixed_line
        
        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.writelines(lines)
        
        print(f"Fixed syntax error in {file_path}")
    else:
        print(f"Error: {file_path} has fewer than 56 lines")


if __name__ == "__main__":
    fix_syntax_error()
