#!/usr/bin/env python3
"""
Test script for the inference CLI tool
"""

import os
import sys
import subprocess
from pathlib import Path


def test_help():
    """Test help functionality"""
    print("Testing help command...")
    try:
        result = subprocess.run(
            [sys.executable, "main2.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        if result.returncode == 0:
            print("✓ Help command works")
            print("Help output preview:")
            print(
                result.stdout[:500] + "..."
                if len(result.stdout) > 500
                else result.stdout
            )
        else:
            print("✗ Help command failed")
            print("Error:", result.stderr)

    except Exception as e:
        print(f"✗ Error running help: {e}")


def test_argument_validation():
    """Test argument validation"""
    print("\nTesting argument validation...")

    # Test missing required arguments
    try:
        result = subprocess.run(
            [sys.executable, "main2.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        if result.returncode != 0 and "required" in result.stderr.lower():
            print("✓ Required argument validation works")
        else:
            print("✗ Required argument validation failed")
            print("Error:", result.stderr)

    except Exception as e:
        print(f"✗ Error testing validation: {e}")


def test_import_structure():
    """Test if all imports work"""
    print("\nTesting import structure...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("."))))
try:
    from inference.main2 import InferenceCLI, create_argument_parser
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")
            """,
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)

    except Exception as e:
        print(f"✗ Error testing imports: {e}")


if __name__ == "__main__":
    print("Testing Inference CLI Tool")
    print("=" * 40)

    test_help()
    test_argument_validation()
    test_import_structure()

    print("\n" + "=" * 40)
    print("Test completed!")
