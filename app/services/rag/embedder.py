import logging
import asyncio
from app.services.llm.clients import ollama_client as client
from app.config import _RETRY_ATTEMPTS, _RETRY_DELAY, EMBED_MODEL_NAME



logger = logging.getLogger(__name__)

async def embed_text(text: str) -> list[float]:
    """Embed a single text string into a vector using Ollama.

    Args:
        text (str): Input text string to embed.

    Raises:
        Exception: If Ollama embedding request fails.

    Returns:
        list[float]: Vector representation of the input text.
    """
    try:
        batch = await client.embed(model=EMBED_MODEL_NAME, input=[text])
        return batch['embeddings'][0]

    except Exception as e:
        print(f"Error in embedding with {EMBED_MODEL_NAME}: {e}")
        raise e


async def embed_batch(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """Embed a list of text strings in batches using Ollama.

    Args:
        texts (list[str]): List of input text strings to embed.
        batch_size (int, optional): Maximum number of texts per batch. Defaults to 20.

    Returns:
        list[list[float]]: List of vector embeddings corresponding to each input text.
    """
    if not texts:
        return []

    results: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = await _embed_batch_retry(batch)
        results.extend(embeddings)

    return results


async def _embed_batch_retry(texts: list[str]) -> list[list[float]]:
    """Embed a single batch of texts with retry attempts on failure.

    Args:
        texts (list[str]): List of texts in the current batch.

    Raises:
        RuntimeError: If all retry attempts fail after maximum attempts.

    Returns:
        list[list[float]]: List of vector embeddings for the batch.
    """
    last_error: Exception | None = None

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            batch = await client.embed(model=EMBED_MODEL_NAME, input=texts)
            return batch['embeddings']

        except Exception as e:
            last_error = e
            logging.warning(
                f"Embedding attemp {attempt}/{_RETRY_ATTEMPTS} failed: {e}"
                + (f" - retrying in {_RETRY_DELAY}s" if attempt < _RETRY_ATTEMPTS else "")
            )
            if attempt < _RETRY_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY)

    raise RuntimeError(
            f"Embedding failed after {_RETRY_ATTEMPTS} attempts: {last_error}"
        )


async def main():
    embeddings = await embed_batch([
        "Hello",
        "Hahahaah"
    ])

    print(len(embeddings))

if __name__ == "__main__":
    asyncio.run(main())