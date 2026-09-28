import os
from dotenv import load_dotenv
from groq import AsyncGroq
from ollama import AsyncClient

load_dotenv()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")
GENERATE_MODEL_NAME = os.getenv("GENERATE_MODEL_NAME", "qwen2.5:1.5b")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_SMALL_MODEL = os.getenv("GROQ_SMALL_MODEL", "qwen/qwen3.8-27b")

ollama_client = AsyncClient(host=OLLAMA_URL)
groq_api_key = os.getenv("GROQ_API_KEY") or "gsk_placeholder"
groq_client = AsyncGroq(api_key=groq_api_key)

class OllamaConnectionError(Exception):
    """Raised when connection to the Ollama service fails."""
    pass

class OllamaModelNotFound(Exception):
    """Raised when the specified model is not found in Ollama."""
    pass