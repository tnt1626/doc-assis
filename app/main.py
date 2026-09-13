import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from app.routers.chat import agent_router
from app.routers.documents import doc_router
from app.routers.sessions import session_router
from app.services.client import ollama_client, EMBED_MODEL_NAME, GENERATE_MODEL_NAME, OLLAMA_URL

logger = logging.getLogger(__name__)

async def pull_ollama_models():
    try:
        logger.info(f"Connecting to Ollama at: {OLLAMA_URL}")
        logger.info(f"Starting to pull Ollama models: {EMBED_MODEL_NAME}, {GENERATE_MODEL_NAME}")
        await ollama_client.pull(EMBED_MODEL_NAME)
        await ollama_client.pull(GENERATE_MODEL_NAME)
        logger.info("Successfully pulled Ollama models.")
    except Exception as e:
        logger.error(f"Failed to pull Ollama models (URL: {OLLAMA_URL}): {e}. Application will continue starting, but Ollama might be unavailable.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(pull_ollama_models())
    yield

app = FastAPI(
    title="Document Q&A API",
    description="Upload documents and query them using natural language.",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(doc_router)
app.include_router(agent_router)
app.include_router(session_router)
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

@app.exception_handler(Exception)
async def global_exception_hanlder(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

