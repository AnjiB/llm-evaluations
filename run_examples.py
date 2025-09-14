#!/usr/bin/env python3
"""
Example Runner for OpenEvals Examples

This script provides an easy way to run all the examples in this repository.
"""

import os
import sys
import argparse
from pathlib import Path

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables and try again.")
        print("Example:")
        print("   export OPENAI_API_KEY='your_api_key_here'")
        return False
    
    print("✅ Environment variables are set correctly")
    return True

def run_example(example_name: str):
    """Run a specific example"""
    examples_dir = Path(__file__).parent / "examples"
    example_file = examples_dir / f"{example_name}.py"
    
    if not example_file.exists():
        print(f"❌ Example '{example_name}' not found")
        print(f"Available examples: {', '.join(get_available_examples())}")
        return False
    
    print(f"🚀 Running {example_name} example...")
    print("=" * 50)
    
    try:
        # Change to examples directory and run the script with Poetry
        import subprocess
        result = subprocess.run([
            "poetry", "run", "python", str(example_file)
        ], cwd=examples_dir, capture_output=False)
        
        if result.returncode == 0:
            print(f"✅ {example_name} completed successfully")
            return True
        else:
            print(f"❌ {example_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {example_name}: {e}")
        return False

def get_available_examples():
    """Get list of available examples"""
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        return []
    
    examples = []
    for file in examples_dir.glob("*.py"):
        if file.name != "__init__.py":
            examples.append(file.stem)
    
    return sorted(examples)

def run_all_examples():
    """Run all available examples"""
    examples = get_available_examples()
    
    if not examples:
        print("❌ No examples found")
        return False
    
    print(f"🚀 Running {len(examples)} examples...")
    print("=" * 50)
    
    results = {}
    for example in examples:
        print(f"\n📝 Running {example}...")
        success = run_example(example)
        results[example] = success
        print()
    
    # Summary
    print("📊 Summary:")
    print("=" * 50)
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    for example, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {example}")
    
    print(f"\nCompleted: {successful}/{total} examples")
    
    return successful == total

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run OpenEvals examples")
    parser.add_argument(
        "example", 
        nargs="?", 
        help="Specific example to run (optional)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available examples"
    )
    parser.add_argument(
        "--check-env", 
        action="store_true", 
        help="Check environment variables"
    )
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # List examples
    if args.list:
        examples = get_available_examples()
        if examples:
            print("Available examples:")
            for example in examples:
                print(f"  - {example}")
        else:
            print("No examples found")
        return
    
    # Check environment only
    if args.check_env:
        return
    
    # Run specific example
    if args.example:
        success = run_example(args.example)
        sys.exit(0 if success else 1)
    
    # Run all examples
    success = run_all_examples()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
