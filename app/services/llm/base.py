from typing import AsyncIterator, Protocol
from app.services.llm import llm_types

class LLMClient(Protocol):
    """Protocol interface defining required LLM streaming and completion operations."""

    async def stream(
        self,
        messages: list[dict[str, str]],
        purpose: llm_types.LLMPurpose
    ) -> AsyncIterator[llm_types.TextDelta | llm_types.ToolCall | llm_types.Usage]:
        """Stream response events from LLM provider asynchronously.

        Args:
            messages (list[dict[str, str]]): List of input message objects.
            purpose (llm_types.LLMPurpose): Purpose category of the LLM request.

        Yields:
            AsyncIterator[llm_types.TextDelta | llm_types.ToolCall | llm_types.Usage]:
            Streamed text tokens, tool call objects, and usage statistics.
        """
        ...

    async def completion(
        self,
        messages: list[dict[str, str]],
        purpose: llm_types.LLMPurpose,
        max_tokens: int = 600
    ) -> str:
        """Execute a non-streaming completion call for background memory tasks.

        Args:
            messages (list[dict[str, str]]): List of input message objects.
            purpose (llm_types.LLMPurpose): Purpose category of the LLM request.
            max_tokens (int, optional): Maximum tokens permitted. Defaults to 600.

        Returns:
            str: Response text content.
        """
        ...