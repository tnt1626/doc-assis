import os
from groq import AsyncGroq
from ollama import AsyncClient
from dotenv import load_dotenv
from app.config import OLLAMA_URL

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or "gsk_placeholder"

ollama_client = AsyncClient(host=OLLAMA_URL)
groq_client = AsyncGroq(api_key=groq_api_key)