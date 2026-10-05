import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MIN_CHARS = 80
BATCH_SIZE = 20
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0 
THRESHOLD = 0.6
PAGE_SIZE = 5000
MAX_CHARS = 500
OVERLAP_SENTENCES = 1
OLLAMA_TIMEOUT = 2
AGENT_DIR = Path(".agent").resolve()
USER_PROFILE_FILE = "USER.md"
SOUL_FILE = "SOUL.md"
CONSOLIDATE_EVERY_N = 20
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_SMALL_MODEL = os.getenv("GROQ_SMALL_MODEL", "qwen/qwen3.8-27b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")