from app.services.llm import groq
from app.services.llm.clients import groq_client
from app.services.agent.tools import TOOLS
from app.config import GROQ_MODEL, GROQ_SMALL_MODEL


groq_llm = groq.GroqClient(
    client=groq_client,
    model=GROQ_MODEL,
    small_model=GROQ_SMALL_MODEL,
    tools=TOOLS
)

class OllamaConnectionError(Exception):
    """Raised when connection to the Ollama service fails."""
    pass

class OllamaModelNotFound(Exception):
    """Raised when the specified model is not found in Ollama."""
    pass