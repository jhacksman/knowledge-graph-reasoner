#!/usr/bin/env python3
"""
Fix script for MockVectorStore implementation issues.

This script implements the required abstract methods in MockVectorStore class in test files.
"""
import os
import re
import asyncio
from pathlib import Path
from typing import List, AsyncIterator


def fix_mock_vector_store():
    """Fix MockVectorStore implementation issues."""
    # Define the paths to the test files
    test_files = [
        Path("tests/test_graph_manager.py"),
        Path("tests/test_query_executor.py")
    ]
    
    for test_file in test_files:
        if not test_file.exists():
            print(f"Error: {test_file} does not exist")
            continue
        
        # Read the current content of the file
        with open(test_file, "r") as file:
            content = file.read()
        
        # Add missing abstract methods to MockVectorStore
        if "class MockVectorStore(BaseVectorStore):" in content:
            # Check if the required methods are already implemented
            missing_methods = []
            if "async def get_all_nodes" not in content:
                missing_methods.append("""    async def get_all_nodes(self) -> AsyncIterator[Node]:
        \"\"\"Get all nodes implementation for MockVectorStore.\"\"\"
        nodes = [
            Node(id="test1", content="Test content 1"),
            Node(id="test2", content="Test content 2")
        ]
        for node in nodes:
            yield node
            await asyncio.sleep(0)  # Allow other coroutines to run""")
            
            if "async def get_all_edges" not in content:
                missing_methods.append("""    async def get_all_edges(self) -> AsyncIterator[Edge]:
        \"\"\"Get all edges implementation for MockVectorStore.\"\"\"
        edges = [
            Edge(source="test1", target="test2", type="related"),
            Edge(source="test2", target="test3", type="similar")
        ]
        for edge in edges:
            yield edge
            await asyncio.sleep(0)  # Allow other coroutines to run""")
            
            if "async def update_node" not in content:
                missing_methods.append("""    async def update_node(self, node: Node) -> None:
        \"\"\"Update node implementation for MockVectorStore.\"\"\"
        # This is a mock method, so no actual implementation needed
        pass""")
            
            if missing_methods:
                # Insert the missing methods at the end of the class
                class_pattern = r"class MockVectorStore\(BaseVectorStore\):.*?(?=\n\n)"
                class_match = re.search(class_pattern, content, re.DOTALL)
                if class_match:
                    class_content = class_match.group(0)
                    new_class_content = class_content + "\n\n" + "\n\n".join(missing_methods)
                    content = content.replace(class_content, new_class_content)
        
        # Save the updated content
        with open(test_file, "w") as file:
            file.write(content)
        
        print(f"Fixed MockVectorStore implementation in {test_file}")


if __name__ == "__main__":
    fix_mock_vector_store()
