"""
LangSmith Integration Examples

This module demonstrates how to integrate OpenEvals with LangSmith
for tracking, monitoring, and analyzing LLM evaluations over time.
"""

import os
from typing import Dict, Any, List
from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT, HALLUCINATION_PROMPT, CORRECTNESS_PROMPT

# Optional pytest import
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# Set your LangSmith API key
os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key_here"
os.environ["LANGSMITH_TRACING"] = "true"

# Import LangSmith testing utilities
try:
    from langsmith import testing as t
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    print("LangSmith not available. Install with: pip install langsmith")
    LANGSMITH_AVAILABLE = False


def example_pytest_integration():
    """
    Example of using OpenEvals with pytest and LangSmith
    """
    if not LANGSMITH_AVAILABLE:
        print("LangSmith not available. Skipping pytest integration example.")
        return
    
    print("=== Pytest + LangSmith Integration Example ===")
    
    # Create evaluators
    conciseness_evaluator = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        feedback_key="conciseness",
        model="openai:o3-mini",
    )
    
    hallucination_evaluator = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )
    
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model="openai:o3-mini",
    )
    
    # Test cases
    test_cases = [
        {
            "name": "concise_answer",
            "inputs": "What is the capital of France?",
            "outputs": "Paris",
            "reference_outputs": "Paris",
            "context": "France is a country in Western Europe with Paris as its capital.",
            "expected_conciseness": True,
            "expected_correctness": True,
            "expected_hallucination": True
        },
        {
            "name": "verbose_answer",
            "inputs": "What is the capital of France?",
            "outputs": "Thanks for asking! The capital of France is Paris. I hope this helps!",
            "reference_outputs": "Paris",
            "context": "France is a country in Western Europe with Paris as its capital.",
            "expected_conciseness": False,
            "expected_correctness": True,
            "expected_hallucination": True
        },
        {
            "name": "incorrect_answer",
            "inputs": "What is the capital of France?",
            "outputs": "London",
            "reference_outputs": "Paris",
            "context": "France is a country in Western Europe with Paris as its capital.",
            "expected_conciseness": True,
            "expected_correctness": False,
            "expected_hallucination": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\nRunning test: {test_case['name']}")
        
        # Log inputs and outputs to LangSmith
        t.log_inputs({"question": test_case['inputs']})
        t.log_outputs({"answer": test_case['outputs']})
        t.log_reference_outputs({"answer": test_case['reference_outputs']})
        
        # Run evaluations
        conciseness_result = conciseness_evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs']
        )
        
        hallucination_result = hallucination_evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
            context=test_case['context']
        )
        
        correctness_result = correctness_evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
            reference_outputs=test_case['reference_outputs']
        )
        
        # Log evaluation results
        t.log_feedback(
            key="conciseness",
            score=conciseness_result['score'],
            comment=conciseness_result['comment']
        )
        
        t.log_feedback(
            key="hallucination",
            score=hallucination_result['score'],
            comment=hallucination_result['comment']
        )
        
        t.log_feedback(
            key="correctness",
            score=correctness_result['score'],
            comment=correctness_result['comment']
        )
        
        print(f"Conciseness: {conciseness_result['score']}")
        print(f"Hallucination: {hallucination_result['score']}")
        print(f"Correctness: {correctness_result['score']}")


def example_langsmith_evaluate_function():
    """
    Example of using LangSmith's evaluate function with OpenEvals
    """
    if not LANGSMITH_AVAILABLE:
        print("LangSmith not available. Skipping evaluate function example.")
        return
    
    print("\n=== LangSmith Evaluate Function Example ===")
    
    # Create evaluators
    conciseness_evaluator = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        feedback_key="conciseness",
        model="openai:o3-mini",
    )
    
    hallucination_evaluator = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )
    
    # Wrapper functions for LangSmith evaluate
    def wrapped_conciseness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        """Wrapper for conciseness evaluator"""
        eval_result = conciseness_evaluator(
            inputs=inputs,
            outputs=outputs,
        )
        return eval_result
    
    def wrapped_hallucination_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        """Wrapper for hallucination evaluator"""
        eval_result = hallucination_evaluator(
            inputs=inputs,
            outputs=outputs,
            context=inputs.get('context', '')
        )
        return eval_result
    
    # Mock LLM function (replace with your actual LLM)
    def mock_llm_function(inputs: dict) -> dict:
        """Mock LLM function - replace with your actual LLM"""
        question = inputs.get('question', '')
        
        # Simple mock responses based on question
        if 'capital' in question.lower() and 'france' in question.lower():
            return {"answer": "Paris"}
        elif 'capital' in question.lower() and 'germany' in question.lower():
            return {"answer": "Berlin"}
        else:
            return {"answer": "I don't know the answer to that question."}
    
    # Create LangSmith client
    client = Client()
    
    # Sample dataset
    dataset_name = "llm-evaluation-dataset"
    
    try:
        # Create dataset
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Sample dataset for LLM evaluation"
        )
        
        # Add examples to dataset
        examples = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": {"answer": "Paris"},
                "reference_outputs": {"answer": "Paris"},
                "context": "France is a country in Western Europe with Paris as its capital."
            },
            {
                "inputs": {"question": "What is the capital of Germany?"},
                "outputs": {"answer": "Berlin"},
                "reference_outputs": {"answer": "Berlin"},
                "context": "Germany is a country in Central Europe with Berlin as its capital."
            },
            {
                "inputs": {"question": "What is the capital of Japan?"},
                "outputs": {"answer": "Tokyo"},
                "reference_outputs": {"answer": "Tokyo"},
                "context": "Japan is an island nation in East Asia with Tokyo as its capital."
            }
        ]
        
        for example in examples:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                reference_outputs=example["reference_outputs"],
                dataset_id=dataset.id
            )
        
        print(f"Created dataset: {dataset_name}")
        
        # Run evaluation
        print("Running evaluation...")
        experiment_results = client.evaluate(
            mock_llm_function,
            data=dataset_name,
            evaluators=[
                wrapped_conciseness_evaluator,
                wrapped_hallucination_evaluator
            ]
        )
        
        print(f"Evaluation completed! Results: {experiment_results}")
        
    except Exception as e:
        print(f"Error with LangSmith evaluation: {e}")


def example_custom_evaluator_with_langsmith():
    """
    Example of custom evaluator with LangSmith integration
    """
    if not LANGSMITH_AVAILABLE:
        print("LangSmith not available. Skipping custom evaluator example.")
        return
    
    print("\n=== Custom Evaluator with LangSmith Example ===")
    
    # Create custom evaluator
    custom_evaluator = create_llm_as_judge(
        prompt="""You are an expert evaluator assessing response quality across multiple dimensions.

<Question>
{inputs}
</Question>

<Response>
{outputs}
</Response>

<Reference>
{reference_outputs}
</Reference>

Evaluate the response on:
1. Accuracy (0-1): How correct is the information?
2. Clarity (0-1): How clear and understandable is the response?
3. Completeness (0-1): How complete is the answer?
4. Helpfulness (0-1): How helpful is the response to the user?

Respond with JSON:
{{
    "accuracy": 0.0-1.0,
    "clarity": 0.0-1.0,
    "completeness": 0.0-1.0,
    "helpfulness": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reasoning": "detailed explanation"
}}""",
        feedback_key="quality",
        model="openai:o3-mini",
        output_schema={
            "type": "object",
            "properties": {
                "accuracy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "clarity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "completeness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "helpfulness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string"}
            },
            "required": ["accuracy", "clarity", "completeness", "helpfulness", "overall_score", "reasoning"]
        }
    )
    
    # Test the custom evaluator
    test_cases = [
        {
            "inputs": "What is machine learning?",
            "outputs": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "reference_outputs": "Machine learning is a method of data analysis that automates analytical model building."
        },
        {
            "inputs": "How do I bake a cake?",
            "outputs": "Mix flour, sugar, eggs, and butter. Bake at 350°F for 30 minutes.",
            "reference_outputs": "Combine dry ingredients, add wet ingredients, mix well, and bake at 350°F for 25-30 minutes until golden brown."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Question: {test_case['inputs']}")
        print(f"Response: {test_case['outputs']}")
        
        # Log to LangSmith
        t.log_inputs({"question": test_case['inputs']})
        t.log_outputs({"answer": test_case['outputs']})
        t.log_reference_outputs({"answer": test_case['reference_outputs']})
        
        # Run evaluation
        result = custom_evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
            reference_outputs=test_case['reference_outputs']
        )
        
        # Log results to LangSmith
        t.log_feedback(
            key="quality",
            score=result['overall_score'],
            comment=result['reasoning']
        )
        
        print(f"Accuracy: {result['accuracy']:.2f}")
        print(f"Clarity: {result['clarity']:.2f}")
        print(f"Completeness: {result['completeness']:.2f}")
        print(f"Helpfulness: {result['helpfulness']:.2f}")
        print(f"Overall Score: {result['overall_score']:.2f}")


def run_langsmith_examples():
    """Run all LangSmith integration examples"""
    print("LangSmith Integration Examples")
    print("=" * 50)
    
    if not LANGSMITH_AVAILABLE:
        print("LangSmith is not available. Please install it with: pip install langsmith")
        return
    
    try:
        example_pytest_integration()
        example_langsmith_evaluate_function()
        example_custom_evaluator_with_langsmith()
        
        print("\n" + "=" * 50)
        print("All LangSmith integration examples completed!")
        print("Check your LangSmith dashboard to view the results.")
        
    except Exception as e:
        print(f"Error running LangSmith examples: {e}")
        print("Make sure you have set your LANGSMITH_API_KEY environment variable.")


if __name__ == "__main__":
    run_langsmith_examples()
