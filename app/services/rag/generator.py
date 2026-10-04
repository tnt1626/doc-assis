import uuid
import ollama
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import GENERATE_MODEL_NAME
from app.services.rag.retriever import retrieve_vec
from app.services.llm.clients import ollama_client as client
from app.services.client import (
    OllamaModelNotFound, 
    OllamaConnectionError,  
    
)


@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(ollama.RequestError)
)
async def generate(
    question: str, 
    document_id: uuid.UUID, 
    top_k: int, 
    db: AsyncSession
):
    """Generate an answer to a user question based on relevant document chunks using Ollama.

    Args:
        question (str): User's natural language question.
        document_id (uuid.UUID): ID of the target document to retrieve context from.
        top_k (int): Number of most relevant document chunks to use.
        db (AsyncSession): Database session for chunk retrieval.

    Raises:
        OllamaConnectionError: If connection to Ollama fails.
        OllamaModelNotFound: If specified generation model is not found in Ollama.
        RuntimeError: If answer generation fails.

    Returns:
        dict: Response containing 'answer' string, total 'token' count, and 'sources' list.
    """
    rows = await retrieve_vec(question, document_id, top_k, db)
    if not rows:
        return {
            "answer": "I don't know based on the provided document.",
            "token": 0,
            "sources": []
        }

    context = ""
    for row in rows:
        chunk = row[0] # chunk
        distance = row[2] # distance
        context += f"Content: {chunk.content}. Distance: {distance}\n"

    system_prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.
Do not use any knowledge outside of the provided context.
If the answer cannot be found in the context, respond exactly with: "I don't know based on the provided document."

Context:
{context}
"""
    try:
        response = await client.chat(
            model=GENERATE_MODEL_NAME,
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]
        )

        return {
            "answer": response.message.content,
            "token": response.prompt_eval_count + response.eval_count,
            "sources": [row[0].content for row in rows]
        }
    except ollama.RequestError:
        raise OllamaConnectionError()
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise OllamaModelNotFound()
        raise
    except Exception as e:
        raise RuntimeError(f"Generate failed: {e}")
