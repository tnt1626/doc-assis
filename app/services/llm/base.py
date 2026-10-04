from typing import AsyncIterator, Protocol
from app.services.llm import llm_types

class LLMClient(Protocol):
    async def stream(
        self,
        messages: list[dict[str, str]],
        purpose: llm_types.LLMPurpose
    ) -> AsyncIterator[llm_types.TextDelta | llm_types.ToolCall | llm_types.Usage]:
        ...

    async def completion(
        self,
        messages: list[dict[str, str]],
        purpose: llm_types.LLMPurpose,
        max_tokens: int = 600
    ) -> str:
        ...