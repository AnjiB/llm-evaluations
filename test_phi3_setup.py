#!/usr/bin/env python3
"""
Test script to verify Phi3 and Ollama setup

This script checks if Ollama is running and Phi3 model is available,
then runs a simple test to verify everything works.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))

from ollama_client import OllamaClient
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

def test_ollama_setup():
    """Test if Ollama is properly set up"""
    print("🔍 Testing Ollama setup...")
    
    client = OllamaClient()
    
    # Check if Ollama is running
    if not client.health_check():
        print("❌ Ollama is not running. Please start Ollama first.")
        print("   Run: ollama serve")
        return False
    
    print("✅ Ollama is running")
    
    # Check if Phi3 model is available
    if not client.is_model_available("phi3"):
        print("❌ Phi3 model not found. Please install it with:")
        print("   ollama pull phi3")
        return False
    
    print("✅ Phi3 model is available")
    
    # Test a simple generation
    try:
        response = client.generate(
            model="phi3:latest",
            prompt="What is 2+2?",
            options={"temperature": 0.1, "max_tokens": 50}
        )
        print(f"✅ Phi3 generation test successful: {response}")
        return True
    except Exception as e:
        print(f"❌ Phi3 generation test failed: {e}")
        return False

def test_openevals_with_phi3():
    """Test OpenEvals with Phi3"""
    print("\n🔍 Testing OpenEvals with Phi3...")
    
    try:
        # Create evaluator
        evaluator = create_llm_as_judge(
            prompt=CORRECTNESS_PROMPT,
            feedback_key="correctness",
            model="ollama:phi3:latest"
        )
        
        # Test evaluation
        result = evaluator(
            inputs="What is 2+2?",
            outputs="4",
            reference_outputs="4"
        )
        
        print(f"✅ OpenEvals with Phi3 test successful")
        print(f"   Score: {result['score']}")
        print(f"   Comment: {result['comment'][:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenEvals with Phi3 test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Phi3 and Ollama Setup Test")
    print("=" * 40)
    
    # Test Ollama setup
    ollama_ok = test_ollama_setup()
    
    if not ollama_ok:
        print("\n❌ Ollama setup test failed. Please fix the issues above.")
        sys.exit(1)
    
    # Test OpenEvals with Phi3
    openevals_ok = test_openevals_with_phi3()
    
    if not openevals_ok:
        print("\n❌ OpenEvals with Phi3 test failed. Please check your setup.")
        sys.exit(1)
    
    print("\n🎉 All tests passed! You're ready to use Phi3 with OpenEvals.")
    print("\nNext steps:")
    print("1. Run: python examples/phi3_real_test.py")
    print("2. Run: python examples/phi3_ollama_test.py")

if __name__ == "__main__":
    main()
