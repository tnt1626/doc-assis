import os
from groq import AsyncGroq
from ollama import AsyncClient
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.config import OLLAMA_URL

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or "gsk_placeholder"

ollama_client = AsyncClient(host=OLLAMA_URL)
groq_client = AsyncGroq(api_key=groq_api_key)
ollama_openai_client = AsyncOpenAI(
    base_url=f"{OLLAMA_URL}/v1",
    api_key="ollama",
)


class OllamaConnectionError(Exception):
    """Raised when connection to the Ollama service fails."""
    pass

class OllamaModelNotFound(Exception):
    """Raised when the specified model is not found in Ollama."""
    pass
