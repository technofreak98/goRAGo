#!/usr/bin/env python3
"""Test runner script for the RAG-ES system."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False


def main():
    """Main test runner function."""
    print("🧪 RAG-ES Test Runner")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"🐍 Using conda environment: {conda_env}")
    else:
        print("⚠️  Warning: Not in a conda environment. Consider using 'conda activate Agentic-RAG'")
    
    # Test commands
    test_commands = [
        # Unit tests
        ("python -m pytest tests/unit/ -v --tb=short", "Unit Tests"),
        
        # Integration tests
        ("python -m pytest tests/integration/ -v --tb=short", "Integration Tests"),
        
        # All tests with coverage
        ("python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html:htmlcov", "All Tests with Coverage"),
        
        # Lint tests
        ("python -m flake8 app/ tests/ --max-line-length=100 --ignore=E203,W503", "Code Linting"),
        
        # Type checking (if mypy is available)
        ("python -m mypy app/ --ignore-missing-imports", "Type Checking"),
    ]
    
    # Run tests
    results = []
    for command, description in test_commands:
        success = run_command(command, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} | Passed: {passed} | Failed: {failed}")
    
    if failed > 0:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
