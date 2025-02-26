#!/usr/bin/env python3
"""
Main script to run all type checking fixes.
"""
import os
import sys

# Add scripts directory to path to import fix modules
sys.path.append(os.path.dirname(__file__))

from fix_relationship_routes import fix_relationship_routes
from fix_return_types import fix_all_return_types
from fix_mock_vector_store import fix_mock_vector_store
from fix_test_constructors import fix_test_constructors
from fix_checkpoint_manager_access import fix_checkpoint_manager_access
from fix_middleware_attributes import fix_middleware_attributes


def run_all_fixes():
    """Run all fix scripts."""
    print("Starting to apply type checking fixes...")
    
    # Apply fixes
    fix_relationship_routes()
    fix_all_return_types()
    fix_mock_vector_store()
    fix_test_constructors()
    fix_checkpoint_manager_access()
    fix_middleware_attributes()
    
    print("\nAll type checking fixes have been applied!")
    print("Please run mypy again to verify the fixes.")


if __name__ == "__main__":
    run_all_fixes()