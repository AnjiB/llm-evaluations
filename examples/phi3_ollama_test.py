"""
Phi3 with Ollama Local LLM Evaluation Test

This module demonstrates how to use OpenEvals with a local Phi3 model
running on Ollama for LLM evaluation without requiring OpenAI API keys.
"""

import os
import json
from typing import Dict, Any, List
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, HALLUCINATION_PROMPT

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
PHI3_MODEL = "phi3:latest"  # or "phi3:3.8b" for specific version

def create_phi3_evaluator(prompt: str, feedback_key: str):
    """
    Create an evaluator using local Phi3 model via Ollama
    """
    return create_llm_as_judge(
        prompt=prompt,
        feedback_key=feedback_key,
        model=f"ollama:{PHI3_MODEL}",
    )

def test_phi3_correctness_evaluation():
    """
    Test correctness evaluation using Phi3 model
    """
    print("=== Phi3 Correctness Evaluation Test ===")
    
    # Create correctness evaluator
    correctness_evaluator = create_phi3_evaluator(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness"
    )
    
    # Test cases
    test_cases = [
        {
            "inputs": "What is the capital of France?",
            "outputs": "Paris",
            "reference_outputs": "Paris",
            "description": "Correct answer"
        },
        {
            "inputs": "What is 15 + 27?",
            "outputs": "42",
            "reference_outputs": "42",
            "description": "Correct mathematical answer"
        },
        {
            "inputs": "What is 15 + 27?",
            "outputs": "41",
            "reference_outputs": "42",
            "description": "Incorrect mathematical answer"
        },
        {
            "inputs": "Who wrote Romeo and Juliet?",
            "outputs": "William Shakespeare",
            "reference_outputs": "William Shakespeare",
            "description": "Correct literary answer"
        },
        {
            "inputs": "Who wrote Romeo and Juliet?",
            "outputs": "Charles Dickens",
            "reference_outputs": "William Shakespeare",
            "description": "Incorrect literary answer"
        }
    ]
    
    print(f"Testing with Phi3 model: {PHI3_MODEL}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print("-" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Question: {test_case['inputs']}")
        print(f"Answer: {test_case['outputs']}")
        print(f"Reference: {test_case['reference_outputs']}")
        
        try:
            result = correctness_evaluator(
                inputs=test_case['inputs'],
                outputs=test_case['outputs'],
                reference_outputs=test_case['reference_outputs']
            )
            
            print(f"Correctness Score: {result['score']}")
            print(f"Reasoning: {result['comment']}")
            
            results.append({
                "test_case": i,
                "description": test_case['description'],
                "score": result['score'],
                "reasoning": result['comment']
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "test_case": i,
                "description": test_case['description'],
                "score": None,
                "error": str(e)
            })
        
        print("-" * 40)
    
    return results

def test_phi3_hallucination_evaluation():
    """
    Test hallucination detection using Phi3 model
    """
    print("\n=== Phi3 Hallucination Detection Test ===")
    
    # Create hallucination evaluator
    hallucination_evaluator = create_phi3_evaluator(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination"
    )
    
    # Test cases with context
    test_cases = [
        {
            "inputs": "Who were the Apollo 11 astronauts?",
            "outputs": "Neil Armstrong, Buzz Aldrin, and Michael Collins.",
            "context": "Neil Armstrong and Buzz Aldrin were the first humans to land on the Moon on July 20, 1969. Michael Collins remained in lunar orbit.",
            "description": "Factually correct answer"
        },
        {
            "inputs": "Who were the Apollo 11 astronauts?",
            "outputs": "Neil Armstrong, Buzz Aldrin, Michael Collins, and Sarah Johnson.",
            "context": "Neil Armstrong and Buzz Aldrin were the first humans to land on the Moon on July 20, 1969. Michael Collins remained in lunar orbit.",
            "description": "Hallucinated answer (Sarah Johnson never existed)"
        },
        {
            "inputs": "What is the population of Tokyo?",
            "outputs": "Tokyo has approximately 14 million people living in the city proper.",
            "context": "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world. The Greater Tokyo Area has over 37 million residents, making it the most populous metropolitan area globally.",
            "description": "Partially accurate but imprecise answer"
        },
        {
            "inputs": "What is the population of Tokyo?",
            "outputs": "Tokyo has 50 million people.",
            "context": "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world. The Greater Tokyo Area has over 37 million residents, making it the most populous metropolitan area globally.",
            "description": "Incorrect population figure"
        }
    ]
    
    print(f"Testing with Phi3 model: {PHI3_MODEL}")
    print("-" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Question: {test_case['inputs']}")
        print(f"Answer: {test_case['outputs']}")
        print(f"Context: {test_case['context'][:100]}...")
        
        try:
            result = hallucination_evaluator(
                inputs=test_case['inputs'],
                outputs=test_case['outputs'],
                context=test_case['context']
            )
            
            print(f"Hallucination Score: {result['score']} (True = No hallucination)")
            print(f"Reasoning: {result['comment']}")
            
            results.append({
                "test_case": i,
                "description": test_case['description'],
                "score": result['score'],
                "reasoning": result['comment']
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "test_case": i,
                "description": test_case['description'],
                "score": None,
                "error": str(e)
            })
        
        print("-" * 40)
    
    return results

def test_phi3_question_generation_and_evaluation():
    """
    Test where we ask Phi3 a question and then evaluate its response
    """
    print("\n=== Phi3 Question Generation and Evaluation Test ===")
    
    # This would require implementing a function to call Phi3 directly
    # For now, we'll simulate some responses that Phi3 might give
    
    # Simulated Phi3 responses (in a real implementation, you'd call Ollama API)
    simulated_responses = [
        {
            "question": "What is the capital of France?",
            "phi3_response": "The capital of France is Paris.",
            "context": "France is a country in Western Europe with Paris as its capital city.",
            "reference_answer": "Paris"
        },
        {
            "question": "Explain quantum computing in simple terms.",
            "phi3_response": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously.",
            "context": "Quantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement to process information.",
            "reference_answer": "Quantum computing uses quantum mechanics to process information with qubits that can be in multiple states simultaneously."
        },
        {
            "question": "What is the largest planet in our solar system?",
            "phi3_response": "Jupiter is the largest planet in our solar system.",
            "context": "Jupiter is the fifth planet from the Sun and the largest in the Solar System.",
            "reference_answer": "Jupiter"
        }
    ]
    
    # Create evaluators
    correctness_evaluator = create_phi3_evaluator(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness"
    )
    
    hallucination_evaluator = create_phi3_evaluator(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination"
    )
    
    print(f"Testing Phi3 responses with evaluation")
    print("-" * 60)
    
    results = []
    
    for i, response_data in enumerate(simulated_responses, 1):
        print(f"\nTest Case {i}:")
        print(f"Question: {response_data['question']}")
        print(f"Phi3 Response: {response_data['phi3_response']}")
        
        # Evaluate correctness
        try:
            correctness_result = correctness_evaluator(
                inputs=response_data['question'],
                outputs=response_data['phi3_response'],
                reference_outputs=response_data['reference_answer']
            )
            print(f"Correctness: {correctness_result['score']}")
        except Exception as e:
            print(f"Correctness Error: {e}")
            correctness_result = {"score": None, "comment": str(e)}
        
        # Evaluate hallucination
        try:
            hallucination_result = hallucination_evaluator(
                inputs=response_data['question'],
                outputs=response_data['phi3_response'],
                context=response_data['context']
            )
            print(f"Hallucination: {hallucination_result['score']} (True = No hallucination)")
        except Exception as e:
            print(f"Hallucination Error: {e}")
            hallucination_result = {"score": None, "comment": str(e)}
        
        results.append({
            "test_case": i,
            "question": response_data['question'],
            "phi3_response": response_data['phi3_response'],
            "correctness": correctness_result['score'],
            "hallucination": hallucination_result['score'],
            "correctness_reasoning": correctness_result.get('comment', ''),
            "hallucination_reasoning": hallucination_result.get('comment', '')
        })
        
        print("-" * 40)
    
    return results

def check_ollama_connection():
    """
    Check if Ollama is running and Phi3 model is available
    """
    try:
        import requests
        
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            phi3_models = [model for model in models if 'phi3' in model.get('name', '').lower()]
            
            if phi3_models:
                print(f"✅ Ollama is running and Phi3 models found:")
                for model in phi3_models:
                    print(f"   - {model['name']}")
                return True
            else:
                print("❌ Ollama is running but no Phi3 models found.")
                print("Please install Phi3 model: ollama pull phi3")
                return False
        else:
            print(f"❌ Ollama is not responding. Status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Please make sure Ollama is running.")
        print("Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def run_phi3_tests():
    """
    Run all Phi3 evaluation tests
    """
    print("Phi3 with Ollama - Local LLM Evaluation Tests")
    print("=" * 60)
    
    # Check Ollama connection first
    if not check_ollama_connection():
        print("\nPlease fix Ollama connection issues before running tests.")
        return
    
    print("\n🚀 Starting Phi3 evaluation tests...")
    
    try:
        # Run correctness tests
        correctness_results = test_phi3_correctness_evaluation()
        
        # Run hallucination tests
        hallucination_results = test_phi3_hallucination_evaluation()
        
        # Run question generation and evaluation tests
        generation_results = test_phi3_question_generation_and_evaluation()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        # Correctness summary
        correct_scores = [r for r in correctness_results if r.get('score') is not None]
        if correct_scores:
            correct_count = sum(1 for r in correct_scores if r['score'])
            print(f"Correctness Tests: {correct_count}/{len(correct_scores)} passed")
        
        # Hallucination summary
        hallucination_scores = [r for r in hallucination_results if r.get('score') is not None]
        if hallucination_scores:
            no_hallucination_count = sum(1 for r in hallucination_scores if r['score'])
            print(f"Hallucination Tests: {no_hallucination_count}/{len(hallucination_scores)} passed (no hallucination)")
        
        # Generation summary
        generation_scores = [r for r in generation_results if r.get('correctness') is not None and r.get('hallucination') is not None]
        if generation_scores:
            correct_gen = sum(1 for r in generation_scores if r['correctness'])
            no_hall_gen = sum(1 for r in generation_scores if r['hallucination'])
            print(f"Generation Tests: {correct_gen}/{len(generation_scores)} correct, {no_hall_gen}/{len(generation_scores)} no hallucination")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        print("Make sure Ollama is running and Phi3 model is installed.")

if __name__ == "__main__":
    run_phi3_tests()
