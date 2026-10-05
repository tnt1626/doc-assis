import httpx
import asyncio
import logging
from sqlalchemy import text
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi import Depends, FastAPI, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app import config
from app.mcp import mcp_server
from app.database import get_db
from app.routers.chat import agent_router
from app.routers.documents import doc_router
from app.routers.sessions import session_router
from app.services.llm.clients import ollama_client

logger = logging.getLogger(__name__)

async def pull_ollama_models():
    try:
        logger.info(f"Connecting to Ollama at: {config.OLLAMA_URL}")
        logger.info(f"Starting to pull Ollama models: {config.EMBED_MODEL_NAME}, {config.OLLAMA_MODEL}")
        await ollama_client.pull(config.EMBED_MODEL_NAME)
        await ollama_client.pull(config.OLLAMA_MODEL)
        logger.info("Successfully pulled Ollama models.")
    except Exception as e:
        logger.error(f"Failed to pull Ollama models (URL: {config.OLLAMA_URL}): {e}. Application will continue starting, but Ollama might be unavailable.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(pull_ollama_models())
    yield

app = FastAPI(
    title="Doc Assistant API",
    description="Upload documents and query them using natural language.",
    version="0.1.0",
    lifespan=lifespan
)


app.include_router(doc_router)
app.include_router(agent_router)
app.include_router(session_router)
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")
app.mount("/mcp", mcp_server.sse_app(), name="mcp")

@app.exception_handler(Exception)
async def global_exception_hanlder(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db)
):
    health_status = {
        "status": "healthy",
        "components": {}
    }

    try:
        await db.execute(text("SELECT 1"))
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["database"] = f"offline: {e}"

    try:
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
            response = await client.get(config.OLLAMA_URL)

        if response.status_code == 200:
            health_status["components"]["ollama"] = "healthy"
        else:
            health_status["status"] = "unhealthy"
            health_status["components"]["ollama"] = f"unhealthy status code: {response.status_code}"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["ollama"] = f"offline: {e}"

    if health_status["status"] == "unhealthy":
        return JSONResponse(status_code=503, content=health_status)

    return health_status
