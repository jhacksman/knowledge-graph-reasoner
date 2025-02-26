"""Script to fix GraphManager initialization."""
import os
from pathlib import Path
import re

def fix_file(file_path):
    """Fix GraphManager initialization in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace GraphManager() with GraphManager(vector_store=MilvusStore())
    content = re.sub(
        r'GraphManager\(\)',
        'GraphManager(vector_store=MilvusStore())',
        content
    )
    
    # Replace Missing positional argument "vector_store" in call to "GraphManager"
    content = re.sub(
        r'GraphManager\(vector_store=None\)',
        'GraphManager(vector_store=MilvusStore())',
        content
    )
    
    # Add import for MilvusStore if not already present
    if 'MilvusStore' in content and 'from src.vector_store.milvus_store import MilvusStore' not in content:
        # Find the last import statement
        import_match = re.search(r'^(from|import).*$', content, re.MULTILINE)
        if import_match:
            last_import_pos = 0
            for match in re.finditer(r'^(from|import).*$', content, re.MULTILINE):
                last_import_pos = match.end()
            
            # Insert the import after the last import statement
            content = content[:last_import_pos] + '\nfrom src.vector_store.milvus_store import MilvusStore' + content[last_import_pos:]
    
    # Write the changes back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed GraphManager initialization in {file_path}")

def main():
    """Main function."""
    # Fix API route files
    api_routes_dir = Path('src/api/routes')
    for file_path in api_routes_dir.glob('*.py'):
        if file_path.is_file():
            fix_file(file_path)
    
    # Fix API main file
    api_main_file = Path('src/api/main.py')
    if api_main_file.exists():
        fix_file(api_main_file)

if __name__ == '__main__':
    main()
