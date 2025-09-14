"""
OpenEvals Built-in Evaluators Examples

This module demonstrates how to use OpenEvals' built-in evaluators for common
LLM evaluation tasks like conciseness, hallucination, and correctness.
"""

import os
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CONCISENESS_PROMPT,
    HALLUCINATION_PROMPT,
    CORRECTNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT
)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

def example_conciseness_evaluation():
    """
    Example: Evaluating response conciseness using OpenEvals built-in evaluator
    """
    print("=== Conciseness Evaluation Example ===")
    
    # Create the conciseness evaluator
    conciseness_evaluator = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        feedback_key="conciseness",
        model="openai:o3-mini",
    )
    
    # Test cases
    test_cases = [
        {
            "inputs": "What is the capital of France?",
            "outputs": "Paris",
            "description": "Perfectly concise answer"
        },
        {
            "inputs": "What is the capital of France?",
            "outputs": "Thanks for asking! The capital of France is Paris. I hope this helps!",
            "description": "Verbose answer with unnecessary pleasantries"
        },
        {
            "inputs": "How does photosynthesis work?",
            "outputs": "Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen through a complex biochemical process involving chlorophyll in their leaves.",
            "description": "Concise but complete scientific explanation"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Input: {test_case['inputs']}")
        print(f"Output: {test_case['outputs']}")
        
        result = conciseness_evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs']
        )
        
        print(f"Conciseness Score: {result['score']}")
        print(f"Reasoning: {result['comment']}")


def example_hallucination_evaluation():
    """
    Example: Evaluating for hallucinations using OpenEvals built-in evaluator
    """
    print("\n=== Hallucination Evaluation Example ===")
    
    # Create the hallucination evaluator
    hallucination_evaluator = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )
    
    # Test cases with context
    test_cases = [
        {
            "inputs": "Who was the first president of the United States?",
            "outputs": "George Washington was the first president of the United States.",
            "context": "George Washington served as the first President of the United States from 1789 to 1797. He was unanimously elected by the Electoral College and is often called the 'Father of His Country'.",
            "description": "Factually correct answer"
        },
        {
            "inputs": "Who was the first president of the United States?",
            "outputs": "Thomas Jefferson was the first president of the United States, serving from 1801 to 1809.",
            "context": "George Washington served as the first President of the United States from 1789 to 1797. He was unanimously elected by the Electoral College and is often called the 'Father of His Country'.",
            "description": "Factually incorrect answer (hallucination)"
        },
        {
            "inputs": "What is the population of Tokyo?",
            "outputs": "Tokyo has approximately 14 million people living in the city proper.",
            "context": "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world. The Greater Tokyo Area has over 37 million residents, making it the most populous metropolitan area globally.",
            "description": "Partially accurate but imprecise answer"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Input: {test_case['inputs']}")
        print(f"Output: {test_case['outputs']}")
        print(f"Context: {test_case['context'][:100]}...")
        
        result = hallucination_evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
            context=test_case['context']
        )
        
        print(f"Hallucination Score: {result['score']} (True = No hallucination)")
        print(f"Reasoning: {result['comment']}")


def example_correctness_evaluation():
    """
    Example: Evaluating correctness using OpenEvals built-in evaluator
    """
    print("\n=== Correctness Evaluation Example ===")
    
   


def example_rag_evaluation():
    """
    Example: Evaluating RAG (Retrieval-Augmented Generation) responses
    """
    print("\n=== RAG Evaluation Example ===")
    
    # Create RAG evaluators
    groundedness_evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        model="openai:o3-mini",
    )
    
    helpfulness_evaluator = create_llm_as_judge(
        prompt=RAG_HELPFULNESS_PROMPT,
        feedback_key="helpfulness",
        model="openai:o3-mini",
    )
    
    # RAG test case
    context = """
    The Python programming language was created by Guido van Rossum and first released in 1991. 
    Python is known for its simple syntax and readability. It supports multiple programming paradigms 
    including procedural, object-oriented, and functional programming. Python 3.0 was released in 2008 
    and is not backward compatible with Python 2.x.
    """
    
    inputs = "When was Python first released and who created it?"
    outputs = "Python was first released in 1991 by Guido van Rossum. It's a programming language known for its simple syntax and readability."
    
    print(f"Input: {inputs}")
    print(f"Output: {outputs}")
    print(f"Context: {context.strip()}")
    
    # Evaluate groundedness (how well the answer is supported by the context)
    groundedness_result = groundedness_evaluator(
        inputs=inputs,
        outputs=outputs,
        context=context
    )
    
    # Evaluate helpfulness (how well the answer addresses the question)
    helpfulness_result = helpfulness_evaluator(
        inputs=inputs,
        outputs=outputs,
        context=context
    )
    
    print(f"\nGroundedness Score: {groundedness_result['score']}")
    print(f"Groundedness Reasoning: {groundedness_result['comment']}")
    
    print(f"\nHelpfulness Score: {helpfulness_result['score']}")
    print(f"Helpfulness Reasoning: {helpfulness_result['comment']}")


def run_all_examples():
    """Run all built-in evaluator examples"""
    print("OpenEvals Built-in Evaluators Examples")
    print("=" * 50)
    
    try:
        example_conciseness_evaluation()
        example_hallucination_evaluation()
        example_correctness_evaluation()
        example_rag_evaluation()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable.")


if __name__ == "__main__":
    run_all_examples()
