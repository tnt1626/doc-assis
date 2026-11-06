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
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# ==========================================
# CRAG (Corrective RAG) Configuration
# ==========================================
CRAG_ENABLED: bool = os.getenv("CRAG_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# Evaluator Settings (Strictly CPU-only)
CRAG_EVALUATOR_MODEL: str = os.getenv("CRAG_EVALUATOR_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CRAG_EVALUATOR_DEVICE: str = "cpu"  # Strictly CPU, no CUDA allocation
CRAG_EVALUATOR_BATCH_SIZE: int = int(os.getenv("CRAG_EVALUATOR_BATCH_SIZE", "16"))
CRAG_EVALUATOR_MAX_LENGTH: int = int(os.getenv("CRAG_EVALUATOR_MAX_LENGTH", "512"))

# Thresholds (Configurable heuristic anchors)
CRAG_THRESHOLD_LOWER: float = float(os.getenv("CRAG_THRESHOLD_LOWER", "0.35"))
CRAG_THRESHOLD_UPPER: float = float(os.getenv("CRAG_THRESHOLD_UPPER", "0.70"))
CRAG_STRIP_THRESHOLD: float = float(os.getenv("CRAG_STRIP_THRESHOLD", "0.40"))

# Knowledge Refinement & Safety Settings
CRAG_PRESERVE_NEIGHBORS: bool = os.getenv("CRAG_PRESERVE_NEIGHBORS", "false").strip().lower() in ("1", "true", "yes", "on")
CRAG_MAX_REFINED_CHARS: int = int(os.getenv("CRAG_MAX_REFINED_CHARS", "3000"))

# Retrieval Scope & External Fallback
CRAG_RETRIEVAL_SCOPE: str = os.getenv("CRAG_RETRIEVAL_SCOPE", "DOCUMENT_ONLY")  # "DOCUMENT_ONLY" | "WEB_ALLOWED"
CRAG_WEB_TIMEOUT: float = float(os.getenv("CRAG_WEB_TIMEOUT", "4.0"))
CRAG_WEB_MAX_RESULTS: int = int(os.getenv("CRAG_WEB_MAX_RESULTS", "3"))

# Telemetry
CRAG_TELEMETRY_ENABLED: bool = os.getenv("CRAG_TELEMETRY_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
