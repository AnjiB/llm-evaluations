# LLM Evaluation Examples with OpenEvals

This repository contains comprehensive examples demonstrating how to use [OpenEvals](https://github.com/langchain-ai/openevals) for evaluating Large Language Model (LLM) outputs. The examples cover both built-in evaluators and custom evaluators for various evaluation metrics, including local LLM evaluation with Ollama and Phi3.

## 🎯 What's Included

- **Built-in Evaluators**: Conciseness, hallucination, correctness, and RAG evaluation
- **Custom Evaluators**: Sentiment analysis, technical accuracy, creativity, safety, comprehensiveness, engagement, and fairness
- **Comprehensive Evaluation Suite**: Multi-metric evaluation pipeline with weighted scoring
- **LangSmith Integration**: Experiment tracking and monitoring with pytest
- **Local LLM Support**: Ollama integration with Phi3 for evaluation without OpenAI API keys
- **Production-Ready Examples**: Real-world evaluation scenarios and best practices

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Poetry (for dependency management)
- Ollama (for local LLM testing)

### Installation

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies using Poetry
poetry install

# Set your OpenAI API key (for OpenAI-based evaluators)
export OPENAI_API_KEY="your_openai_api_key_here"

# Optional: Set LangSmith API key for tracking
export LANGSMITH_API_KEY="your_langsmith_api_key_here"
export LANGSMITH_TRACING="true"
```

### Local LLM Setup with Ollama (Alternative to OpenAI)

If you don't have an OpenAI API key or want to use local models, you can use Ollama with Phi3:

#### 1. Install Ollama

**Option A: Desktop Application**
1. Download Ollama Desktop from [https://ollama.ai/download](https://ollama.ai/download)
2. Install and run the application
3. The Ollama server will start automatically

**Option B: Command Line**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows (PowerShell)
winget install Ollama.Ollama

# Or download from https://ollama.ai/download
```

#### 2. Install Phi3 Model

```bash
# Install Phi3 model (choose one)
ollama pull phi3:latest          # Latest version
ollama pull phi3:3.8b           # Specific version (3.8B parameters)
ollama pull phi3:14b            # Larger version (14B parameters)
```

#### 3. Start Ollama Server

```bash
# Start Ollama server (if not using desktop app)
ollama serve
```

#### 4. Verify Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list

# Test Phi3
ollama run phi3 "What is the capital of France?"
```

### Running Examples

#### Option 1: Use the Example Runner (Recommended)
```bash
# List all available examples
poetry run python run_examples.py --list

# Run all examples
poetry run python run_examples.py

# Run specific examples
poetry run python run_examples.py builtin_evaluators
poetry run python run_examples.py custom_evaluators
poetry run python run_examples.py comprehensive_evaluation_suite
poetry run python run_examples.py langsmith_integration
poetry run python run_examples.py phi3_real_test
```

#### Option 2: Run Individual Examples
```bash
# Built-in evaluators
poetry run python examples/builtin_evaluators.py

# Custom evaluators (6 different metrics)
poetry run python examples/custom_evaluators.py

# Comprehensive evaluation suite
poetry run python examples/comprehensive_evaluation_suite.py

# LangSmith integration
poetry run python examples/langsmith_integration.py

# Local LLM with Ollama/Phi3
poetry run python examples/phi3_ollama_test.py
poetry run python examples/phi3_real_test.py
poetry run python examples/ollama_client.py

# Test setup
poetry run python test_phi3_setup.py
```

## 📚 Examples Overview

### 1. Built-in Evaluators

Demonstrates how to use OpenEvals' built-in evaluators:

- **Conciseness**: Evaluates how concise and to-the-point responses are
- **Hallucination**: Detects when LLM outputs contain unsupported or false information
- **Correctness**: Compares outputs against reference answers
- **RAG Evaluation**: Evaluates Retrieval-Augmented Generation responses

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT

# Create evaluator
conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini",
)

# Evaluate response
result = conciseness_evaluator(
    inputs="What is the capital of France?",
    outputs="Paris"
)
print(f"Score: {result['score']}")
print(f"Comment: {result['comment']}")
```

### 2. Custom Evaluators

Shows how to create custom evaluators for specific needs with 6 different evaluation metrics:

- **Sentiment Analysis**: Evaluates emotional tone and sentiment with confidence scoring
- **Technical Accuracy**: Assesses correctness of technical content with detailed feedback
- **Creativity**: Measures originality and creative thinking across multiple dimensions
- **Safety**: Evaluates content safety and appropriateness with risk assessment
- **Comprehensiveness**: Measures thoroughness of responses with coverage analysis
- **Engagement**: Assesses how engaging content is with improvement suggestions
- **Fairness**: Detects bias and ensures impartial treatment across different groups

```python
# Custom sentiment evaluator
sentiment_prompt = """Analyze the sentiment of the given text...

<Text>
{outputs}
</Text>

Respond with JSON:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}"""

sentiment_evaluator = create_llm_as_judge(
    prompt=sentiment_prompt,
    feedback_key="sentiment",
    model="openai:o3-mini",
    output_schema={
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"}
        },
        "required": ["sentiment", "confidence", "reasoning"]
    }
)
```

### 3. Comprehensive Evaluation Suite

A complete evaluation pipeline that combines multiple evaluators:

- **Multi-metric Evaluation**: Runs multiple evaluators simultaneously
- **Weighted Scoring**: Calculates overall scores with custom weights
- **Batch Processing**: Evaluates multiple samples efficiently
- **Detailed Reporting**: Provides comprehensive evaluation reports

```python
from examples.comprehensive_evaluation_suite import ComprehensiveEvaluator

# Initialize evaluator
evaluator = ComprehensiveEvaluator()

# Run comprehensive evaluation
result = evaluator.evaluate_comprehensive(
    inputs="What is machine learning?",
    outputs="Machine learning is a subset of AI...",
    context="Machine learning is a method of data analysis...",
    reference_outputs="Machine learning enables computers to learn from data."
)

print(f"Overall Score: {result['overall_score']:.2f}")
```

### 4. LangSmith Integration

Demonstrates integration with LangSmith for tracking and monitoring:

- **Pytest Integration**: Using OpenEvals with pytest and LangSmith
- **Evaluate Function**: Using LangSmith's evaluate function
- **Custom Evaluators**: Creating custom evaluators with LangSmith tracking
- **Experiment Tracking**: Monitoring evaluation results over time

```python
import pytest
from langsmith import testing as t
from openevals.llm import create_llm_as_judge

@pytest.mark.langsmith
def test_conciseness():
    evaluator = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        feedback_key="conciseness",
        model="openai:o3-mini",
    )
    
    t.log_inputs({"question": "What is the capital of France?"})
    t.log_outputs({"answer": "Paris"})
    
    result = evaluator(
        inputs="What is the capital of France?",
        outputs="Paris"
    )
    
    t.log_feedback(
        key="conciseness",
        score=result['score'],
        comment=result['comment']
    )
```

### 5. Local LLM with Ollama

Demonstrates evaluation using local LLM models without requiring OpenAI API keys:

- **Ollama Integration**: Using Ollama to run local Phi3 model
- **Real Question Testing**: Asking actual questions to Phi3 and evaluating responses
- **Correctness & Hallucination**: Testing both correctness and hallucination detection
- **Local Evaluation**: Complete evaluation pipeline using local models
- **Ollama Client**: Helper class for easy Ollama API interaction
- **Setup Testing**: Comprehensive setup verification and testing

```python
from ollama_client import OllamaClient
from openevals.llm import create_llm_as_judge

# Initialize Ollama client
client = OllamaClient("http://localhost:11434")

# Ask Phi3 a question
response = client.generate(
    model="phi3:latest",
    prompt="What is the capital of France?"
)

# Create evaluator using local Phi3
evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="ollama:phi3:latest"
)

# Evaluate the response
result = evaluator(
    inputs="What is the capital of France?",
    outputs=response,
    reference_outputs="Paris"
)
```

## 📁 Project Structure

```
llm-evaluations/
├── examples/                          # All example scripts
│   ├── builtin_evaluators.py         # Built-in evaluator examples
│   ├── custom_evaluators.py          # Custom evaluator examples (6 metrics)
│   ├── comprehensive_evaluation_suite.py  # Multi-metric evaluation pipeline
│   ├── langsmith_integration.py      # LangSmith integration examples
│   ├── phi3_ollama_test.py          # Phi3 test with predefined cases
│   ├── phi3_real_test.py            # Real Phi3 question testing
│   ├── ollama_client.py             # Ollama API client helper
│   └── profanity_evaluator.py       # Profanity detection evaluator
├── openevals/                        # OpenEvals framework (cloned)
├── run_examples.py                   # Example runner script
├── test_phi3_setup.py               # Phi3 setup verification
├── phi3_evaluation_results.json     # Sample evaluation results
├── PHI3_SETUP_GUIDE.md              # Detailed Phi3 setup guide
├── MEDIUM_ARTICLE_SUMMARY.md        # Article content summary
├── pyproject.toml                    # Poetry configuration
└── README.md                         # This file
```

## 🎯 Evaluation Metrics Explained

### Built-in Metrics

1. **Conciseness**: Measures how brief and direct responses are
   - Penalizes unnecessary pleasantries and verbose explanations
   - Rewards direct, essential information

2. **Hallucination**: Detects false or unsupported information
   - Compares outputs against provided context
   - Identifies fabricated facts or details

3. **Correctness**: Compares outputs against reference answers
   - Measures factual accuracy
   - Handles different notations and formats

4. **RAG Metrics**: Specialized for Retrieval-Augmented Generation
   - **Groundedness**: How well answers are supported by context
   - **Helpfulness**: How well answers address the question

### Custom Metrics

1. **Sentiment Analysis**: Evaluates emotional tone
2. **Technical Accuracy**: Assesses correctness of technical content
3. **Creativity**: Measures originality and innovation
4. **Safety**: Evaluates content appropriateness
5. **Comprehensiveness**: Measures thoroughness
6. **Engagement**: Assesses how engaging content is

## 🔧 Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your_openai_api_key_here"

# Optional for LangSmith integration
export LANGSMITH_API_KEY="your_langsmith_api_key_here"
export LANGSMITH_TRACING="true"
```

### Model Configuration

You can use different models for evaluation:

```python
# Using different OpenAI models
evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    model="openai:gpt-4o-mini",  # or "openai:o3-mini"
)

# Using other providers
evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    model="anthropic:claude-3-haiku-20240307",
)
```

## 📊 Best Practices

### 1. Choose Appropriate Metrics
- Select metrics that align with your use case
- Consider the trade-offs between different evaluation approaches
- Use multiple metrics for comprehensive assessment

### 2. Provide Good Context
- Include relevant context for hallucination detection
- Provide reference outputs for correctness evaluation
- Use clear, specific prompts

### 3. Handle Edge Cases
- Implement error handling for evaluation failures
- Consider fallback strategies for unreliable evaluators
- Validate evaluation results

### 4. Monitor and Iterate
- Track evaluation results over time
- Use LangSmith for experiment tracking
- Continuously improve evaluation prompts


## 📈 Performance Tips

1. **Batch Processing**: Evaluate multiple samples together
2. **Model Selection**: Use appropriate models for your needs
3. **Caching**: Cache evaluation results when possible
4. **Parallel Processing**: Run independent evaluations in parallel

## 🔍 Code Quality & Structure

This project demonstrates excellent practices for LLM evaluation:

### ✅ Strengths
- **Modular Design**: Each evaluator is self-contained with clear interfaces
- **Comprehensive Error Handling**: Graceful failure handling with informative messages
- **Structured Outputs**: JSON schema validation for consistent evaluation results
- **Real-world Examples**: Practical test cases covering various domains
- **Local LLM Support**: Complete offline evaluation capability with Ollama
- **Production Ready**: Multi-metric evaluation pipelines with weighted scoring
- **Documentation**: Extensive inline documentation and usage examples

### 🏗️ Architecture
- **Separation of Concerns**: Clear separation between evaluators, clients, and utilities
- **Extensibility**: Easy to add new custom evaluators following established patterns
- **Configuration**: Environment-based configuration with sensible defaults
- **Testing**: Comprehensive setup verification and example testing

### 📊 Evaluation Coverage
- **Built-in Metrics**: 4 core evaluation types (conciseness, hallucination, correctness, RAG)
- **Custom Metrics**: 7 specialized evaluators for domain-specific needs
- **Multi-dimensional**: Each evaluator provides detailed reasoning and scoring
- **Batch Processing**: Support for evaluating multiple samples efficiently


## 🙏 Acknowledgments

- [OpenEvals](https://github.com/langchain-ai/openevals) for the evaluation framework
- [LangSmith](https://smith.langchain.com/) for experiment tracking
- [LangChain](https://github.com/langchain-ai/langchain) for the underlying infrastructure


## 📚 Additional Resources

- [OpenEvals Documentation](https://github.com/langchain-ai/openevals)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [LLM Evaluation Best Practices](https://docs.smith.langchain.com/evaluation/best-practices)
- [Ollama Documentation](https://ollama.ai/docs)
- [Phi3 Model Information](https://huggingface.co/microsoft/Phi-3)