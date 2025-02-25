"""Script to fix missing imports in the FastAPI interface."""
import os
from pathlib import Path

def fix_file(file_path):
    """Fix missing imports in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix imports in routes/queries.py
    if 'src/api/routes/queries.py' in str(file_path):
        if 'from fastapi import status' not in content:
            content = content.replace(
                'from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query',
                'from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status',
                1
            )
    
    # Fix imports in routes/expansion.py
    if 'src/api/routes/expansion.py' in str(file_path):
        if 'from fastapi import status' not in content:
            content = content.replace(
                'from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks',
                'from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status',
                1
            )
    
    # Fix imports in routes/metrics.py
    if 'src/api/routes/metrics.py' in str(file_path):
        if 'from typing import Collection' not in content:
            content = content.replace(
                'from typing import List, Dict, Any, Optional',
                'from typing import List, Dict, Any, Optional, Collection',
                1
            )
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed imports in {file_path}")

def main():
    """Main function."""
    # Fix API route files
    api_routes_dir = Path('src/api/routes')
    for file_path in api_routes_dir.glob('*.py'):
        if file_path.is_file():
            fix_file(file_path)

if __name__ == '__main__':
    main()
