# Phi3 with Ollama Setup Guide

This guide shows you how to set up and use Phi3 with Ollama for local LLM evaluation using OpenEvals, without requiring OpenAI API keys.

## 🚀 Quick Setup

### 1. Install Ollama

**Option A: Desktop Application (Recommended)**
1. Go to [https://ollama.ai/download](https://ollama.ai/download)
2. Download and install Ollama Desktop
3. Launch the application - Ollama server starts automatically

**Option B: Command Line**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows (PowerShell)
winget install Ollama.Ollama

# Or download from https://ollama.ai/download
```

### 2. Install Phi3 Model

```bash
# Install Phi3 model (choose one based on your system)
ollama pull phi3:latest          # Latest version (recommended)
ollama pull phi3:3.8b           # 3.8B parameters (faster, less memory)
ollama pull phi3:14b            # 14B parameters (better quality, more memory)
```

### 3. Verify Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list

# Test Phi3 directly
ollama run phi3 "What is the capital of France?"
```

## 🧪 Testing Your Setup

Run the setup test to verify everything is working:

```bash
poetry run python test_phi3_setup.py
```

This will:
- ✅ Check if Ollama is running
- ✅ Verify Phi3 model is available
- ✅ Test Phi3 generation
- ✅ Test OpenEvals integration with Phi3

## 📝 Running the Examples

### 1. Basic Phi3 Test

```bash
poetry run python examples/phi3_ollama_test.py
```

This runs predefined test cases with Phi3 evaluation.

### 2. Real Question Testing

```bash
poetry run python examples/phi3_real_test.py
```

This asks real questions to Phi3 and evaluates the responses for:
- **Correctness**: Is the answer factually correct?
- **Hallucination**: Does the answer contain made-up information?

### 3. Ollama Client Test

```bash
poetry run python examples/ollama_client.py
```

This tests the basic Ollama client functionality.

## 🔧 Configuration

### Model Selection

You can use different Phi3 models by changing the model name:

```python
# In your test files, change this line:
PHI3_MODEL = "phi3:latest"  # or "phi3:3.8b", "phi3:14b"
```

### Ollama Server URL

If Ollama is running on a different port or host:

```python
# In ollama_client.py, change this:
OLLAMA_BASE_URL = "http://localhost:11434"  # Default
# or
OLLAMA_BASE_URL = "http://your-server:11434"
```

## 📊 Understanding the Results

### Correctness Evaluation
- **True**: The answer is factually correct
- **False**: The answer is incorrect
- **Reasoning**: Explanation of why the answer is correct/incorrect

### Hallucination Detection
- **True**: No hallucination detected (answer is grounded in context)
- **False**: Hallucination detected (answer contains unsupported information)
- **Reasoning**: Explanation of what was detected

### Example Output

```
Test 1: Geography
Question: What is the capital of France?
Phi3 Response: The capital of France is Paris.
Reference: Paris
✅ Correctness: True
✅ Hallucination: True (True = No hallucination)
```

## 🛠️ Troubleshooting

### Ollama Not Running
```
❌ Ollama is not running. Please start Ollama first.
```
**Solution**: Start Ollama with `ollama serve` or launch the desktop app.

### Phi3 Model Not Found
```
❌ Phi3 model not found. Please install it with: ollama pull phi3
```
**Solution**: Install Phi3 with `ollama pull phi3:latest`

### Connection Error
```
❌ Cannot connect to Ollama. Please make sure Ollama is running.
```
**Solution**: 
1. Check if Ollama is running: `curl http://localhost:11434/api/tags`
2. Verify the URL in your code
3. Check firewall settings

### Evaluation Error
```
❌ Error in evaluation: ...
```
**Solution**: 
1. Check if Phi3 model is properly loaded
2. Verify OpenEvals installation
3. Check the prompt format

## 💡 Tips for Better Results

### 1. Model Selection
- **phi3:3.8b**: Faster, good for testing
- **phi3:latest**: Balanced performance and quality
- **phi3:14b**: Better quality, requires more memory

### 2. Temperature Settings
```python
# Lower temperature for more consistent results
response = client.generate(
    model="phi3:latest",
    prompt=question,
    options={"temperature": 0.1}  # Lower = more consistent
)
```

### 3. Context Quality
Provide good context for hallucination detection:
```python
context = "France is a country in Western Europe with Paris as its capital city."
```

## 🔄 Integration with Your Workflow

### Using in Your Own Code

```python
from ollama_client import OllamaClient
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

# Initialize
client = OllamaClient()
evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="ollama:phi3:latest"
)

# Ask question
response = client.generate(
    model="phi3:latest",
    prompt="Your question here"
)

# Evaluate
result = evaluator(
    inputs="Your question here",
    outputs=response,
    reference_outputs="Expected answer"
)

print(f"Score: {result['score']}")
print(f"Reasoning: {result['comment']}")
```

### Batch Processing

```python
questions = [
    "What is the capital of France?",
    "What is 2+2?",
    "Who wrote Romeo and Juliet?"
]

for question in questions:
    response = client.generate(model="phi3:latest", prompt=question)
    # Evaluate response...
```

## 📈 Performance Considerations

### Memory Usage
- **phi3:3.8b**: ~2.5GB RAM
- **phi3:latest**: ~4GB RAM  
- **phi3:14b**: ~8GB RAM

### Speed
- **phi3:3.8b**: Fastest
- **phi3:latest**: Medium
- **phi3:14b**: Slowest but best quality

### GPU Acceleration
If you have a compatible GPU, Ollama will automatically use it for faster inference.

## 🎯 Next Steps

1. **Run the tests**: Start with `python test_phi3_setup.py`
2. **Try the examples**: Run `python examples/phi3_real_test.py`
3. **Customize**: Modify the test questions for your use case
4. **Scale up**: Use the patterns in your own applications

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Phi3 Model Information](https://huggingface.co/microsoft/Phi-3)
- [OpenEvals Documentation](https://github.com/langchain-ai/openevals)

---

**Happy evaluating! 🚀**
