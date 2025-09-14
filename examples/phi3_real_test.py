"""
Real Phi3 with Ollama Test

This module demonstrates how to use OpenEvals with a real Phi3 model
running on Ollama, where we ask Phi3 questions and then evaluate its responses.
"""

import os
import json
from typing import Dict, Any, List
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, HALLUCINATION_PROMPT
from ollama_client import OllamaClient

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

def ask_phi3_question(client: OllamaClient, question: str) -> str:
    """
    Ask a question to Phi3 and get its response
    
    Args:
        client: OllamaClient instance
        question: The question to ask
    
    Returns:
        Phi3's response
    """
    try:
        response = client.generate(
            model=PHI3_MODEL,
            prompt=question,
            options={
                "temperature": 0.7,
                "max_tokens": 200
            }
        )
        return response.strip()
    except Exception as e:
        print(f"Error asking Phi3: {e}")
        return f"Error: {e}"

def test_phi3_questions_and_evaluation():
    """
    Test where we ask Phi3 real questions and evaluate its responses
    """
    print("=== Real Phi3 Question and Evaluation Test ===")
    
    # Initialize Ollama client
    client = OllamaClient(OLLAMA_BASE_URL)
    
    # Check if Ollama is running
    if not client.health_check():
        print("❌ Ollama is not running. Please start Ollama first.")
        print("Run: ollama serve")
        return
    
    # Check if Phi3 model is available
    if not client.is_model_available("phi3"):
        print("❌ Phi3 model not found. Please install it with: ollama pull phi3")
        return
    
    print("✅ Ollama is running and Phi3 model is available")
    
    # Create evaluators
    correctness_evaluator = create_phi3_evaluator(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness"
    )
    
    hallucination_evaluator = create_phi3_evaluator(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination"
    )
    
    # Test questions with expected answers and context
    test_questions = [
        {
            "question": "What is the capital of France?",
            "reference_answer": "Paris",
            "context": "France is a country in Western Europe with Paris as its capital city.",
            "category": "Geography"
        },
        {
            "question": "What is 15 + 27?",
            "reference_answer": "42",
            "context": "This is a simple arithmetic addition problem.",
            "category": "Mathematics"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "reference_answer": "William Shakespeare",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
            "category": "Literature"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "reference_answer": "Jupiter",
            "context": "Jupiter is the fifth planet from the Sun and the largest in the Solar System.",
            "category": "Astronomy"
        },
        {
            "question": "Explain photosynthesis in simple terms.",
            "reference_answer": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "context": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
            "category": "Biology"
        }
    ]
    
    print(f"\n🧪 Testing {len(test_questions)} questions with Phi3...")
    print("-" * 80)
    
    results = []
    
    for i, test_data in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {test_data['category']}")
        print(f"Question: {test_data['question']}")
        
        # Ask Phi3 the question
        print("🤖 Asking Phi3...")
        phi3_response = ask_phi3_question(client, test_data['question'])
        print(f"Phi3 Response: {phi3_response}")
        
        # Evaluate correctness
        print("🔍 Evaluating correctness...")
        try:
            correctness_result = correctness_evaluator(
                inputs=test_data['question'],
                outputs=phi3_response,
                reference_outputs=test_data['reference_answer']
            )
            correctness_score = correctness_result['score']
            correctness_reasoning = correctness_result['comment']
            print(f"✅ Correctness: {correctness_score}")
        except Exception as e:
            print(f"❌ Correctness Error: {e}")
            correctness_score = None
            correctness_reasoning = str(e)
        
        # Evaluate hallucination
        print("🔍 Evaluating hallucination...")
        try:
            hallucination_result = hallucination_evaluator(
                inputs=test_data['question'],
                outputs=phi3_response,
                context=test_data['context']
            )
            hallucination_score = hallucination_result['score']
            hallucination_reasoning = hallucination_result['comment']
            print(f"✅ Hallucination: {hallucination_score} (True = No hallucination)")
        except Exception as e:
            print(f"❌ Hallucination Error: {e}")
            hallucination_score = None
            hallucination_reasoning = str(e)
        
        # Store results
        result = {
            "test_number": i,
            "category": test_data['category'],
            "question": test_data['question'],
            "phi3_response": phi3_response,
            "reference_answer": test_data['reference_answer'],
            "correctness_score": correctness_score,
            "hallucination_score": hallucination_score,
            "correctness_reasoning": correctness_reasoning,
            "hallucination_reasoning": hallucination_reasoning
        }
        results.append(result)
        
        print("-" * 60)
    
    return results

def print_evaluation_summary(results: List[Dict[str, Any]]):
    """
    Print a summary of the evaluation results
    """
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    correctness_scores = [r for r in results if r.get('correctness_score') is not None]
    hallucination_scores = [r for r in results if r.get('hallucination_score') is not None]
    
    # Correctness summary
    if correctness_scores:
        correct_count = sum(1 for r in correctness_scores if r['correctness_score'])
        print(f"✅ Correctness: {correct_count}/{len(correctness_scores)} tests passed")
    else:
        print("❌ Correctness: No valid scores")
    
    # Hallucination summary
    if hallucination_scores:
        no_hallucination_count = sum(1 for r in hallucination_scores if r['hallucination_score'])
        print(f"✅ Hallucination: {no_hallucination_count}/{len(hallucination_scores)} tests passed (no hallucination)")
    else:
        print("❌ Hallucination: No valid scores")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 80)
    
    for result in results:
        print(f"\nTest {result['test_number']}: {result['category']}")
        print(f"Question: {result['question']}")
        print(f"Phi3 Response: {result['phi3_response'][:100]}{'...' if len(result['phi3_response']) > 100 else ''}")
        print(f"Reference: {result['reference_answer']}")
        print(f"Correctness: {result['correctness_score']}")
        print(f"Hallucination: {result['hallucination_score']}")
    
    # Save results to file
    with open("phi3_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: phi3_evaluation_results.json")

def run_phi3_real_test():
    """
    Run the complete Phi3 real test
    """
    print("Phi3 with Ollama - Real Question and Evaluation Test")
    print("=" * 80)
    
    try:
        # Run the test
        results = test_phi3_questions_and_evaluation()
        
        if results:
            # Print summary
            print_evaluation_summary(results)
            print("\n🎉 Test completed successfully!")
        else:
            print("\n❌ No results generated. Check Ollama setup.")
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        print("Make sure Ollama is running and Phi3 model is installed.")

if __name__ == "__main__":
    run_phi3_real_test()
