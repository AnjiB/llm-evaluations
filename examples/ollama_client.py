"""
Ollama Client Helper

This module provides a simple client to interact with Ollama API
for calling local LLM models like Phi3.
"""

import requests
import json
from typing import Dict, Any, Optional

class OllamaClient:
    """Simple client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
                     If using Ollama desktop app, this should be the default.
                     If running on different port, update accordingly.
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """
        Generate a response from the specified model
        
        Args:
            model: The model name (e.g., "phi3:latest")
            prompt: The input prompt
            **kwargs: Additional parameters for the generation
        
        Returns:
            The generated response text
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
    
    def chat(self, model: str, messages: list, **kwargs) -> str:
        """
        Chat with the model using a conversation format
        
        Args:
            model: The model name (e.g., "phi3:latest")
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the generation
        
        Returns:
            The generated response text
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
    
    def list_models(self) -> list:
        """
        List available models
        
        Returns:
            List of available models
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get("models", [])
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error listing models: {e}")
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available
        
        Args:
            model_name: The model name to check
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            models = self.list_models()
            return any(model_name in model.get("name", "") for model in models)
        except:
            return False
    
    def health_check(self) -> bool:
        """
        Check if Ollama is running and accessible
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

def test_phi3_with_ollama():
    """
    Test function to demonstrate using Phi3 with Ollama
    """
    print("Testing Phi3 with Ollama...")
    print(f"Checking Ollama at: http://localhost:11434")
    
    # Initialize client
    client = OllamaClient()
    
    # Check if Ollama is running
    if not client.health_check():
        print("❌ Ollama is not running. Please start Ollama first.")
        print("💡 If you're using Ollama desktop app, make sure it's running.")
        print("💡 You can also try: ollama serve")
        return
    
    # Check if Phi3 model is available
    if not client.is_model_available("phi3"):
        print("❌ Phi3 model not found. Please install it with: ollama pull phi3")
        return
    
    print("✅ Ollama is running and Phi3 model is available")
    
    # Test simple generation
    print("\n🧪 Testing simple generation...")
    try:
        response = client.generate(
            model="phi3:latest",
            prompt="What is the capital of France?",
            options={
                "temperature": 0.7,
                "max_tokens": 100
            }
        )
        print(f"Question: What is the capital of France?")
        print(f"Phi3 Response: {response}")
    except Exception as e:
        print(f"Error in generation: {e}")
    
    # Test chat format
    print("\n💬 Testing chat format...")
    try:
        messages = [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        response = client.chat(
            model="phi3:latest",
            messages=messages,
            options={
                "temperature": 0.7,
                "max_tokens": 200
            }
        )
        print(f"Question: Explain quantum computing in simple terms.")
        print(f"Phi3 Response: {response}")
    except Exception as e:
        print(f"Error in chat: {e}")

if __name__ == "__main__":
    test_phi3_with_ollama()
