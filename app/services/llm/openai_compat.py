import logging
from typing import AsyncIterator
from app.services.llm import llm_types

logger = logging.getLogger(__name__)

class OpenAICompatClient:
    """LLM client implementation for OpenAI-compatible APIs adhering to LLMClient protocol."""

    def __init__(
        self, 
        client, 
        model, 
        small_model, 
        tools: list[dict], 
        provider_options: dict | None = None
    ):
        """Initialize OpenAICompatClient with OpenAI SDK client, model names, and tool schemas."""
        self.client             = client
        self.model              = model
        self.small_model        = small_model
        self.tools              = tools
        self.provider_options   = provider_options or {}

    async def stream(
        self, 
        messages: list[dict[str, str]],
        purpose: llm_types.LLMPurpose
    ) -> AsyncIterator[llm_types.TextDelta | llm_types.ToolCall | llm_types.Usage]:
        """Stream response events from OpenAI-compatible API asynchronously.

        Args:
            messages (list[dict[str, str]]): List of conversation message objects.
            purpose (llm_types.LLMPurpose): Category/purpose of the LLM call for token tracking.

        Yields:
            AsyncIterator[llm_types.TextDelta | llm_types.ToolCall | llm_types.Usage]:
            Streamed text deltas, aggregated tool call objects, and final usage statistics.

        Raises:
            RuntimeError: Raised when streaming generation fails.
        """
        accumulated_tc: dict = {}
        usage = None

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                stream=True,
                **self.provider_options,
            )

        except Exception as e:
            logger.error(f"[TOKENS] Streaming failed for purpose={purpose.value}: {e}")
            raise RuntimeError(f"Streaming failed: {e}")

        async for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
                usage = llm_types.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )

                logger.info(f"[TOKENS] purpose={purpose.value} model={self.model} prompt={prompt_tokens} completion={completion_tokens}")

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if getattr(delta, "content", None) is not None:
                piece_content = delta.content

                yield llm_types.TextDelta(text=piece_content)

            if getattr(delta, "tool_calls", None) is not None:
                for tc in delta.tool_calls:
                    index = tc.index
    
                    if index not in accumulated_tc:
                        accumulated_tc[index] = {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": ""
                        }
    
                    if tc.function.arguments:
                        accumulated_tc[index]["arguments"] += tc.function.arguments

        for tc in accumulated_tc.values():
            yield llm_types.ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=tc["arguments"]
            )

        if usage is None:
            logger.warning(f"[TOKENS] purpose={purpose.value} Usage statistics were missing or None.")

        yield usage

        return


    async def completion(
        self, 
        messages: list[dict[str, str]], 
        purpose: llm_types.LLMPurpose,
        max_tokens: int = 600
    ) -> str:
        """Execute a non-streaming completion call using the small model.

        Args:
            messages (list[dict[str, str]]): List of conversation message objects.
            purpose (llm_types.LLMPurpose): Category/purpose of the LLM call for token tracking.
            max_tokens (int, optional): Maximum response tokens permitted. Defaults to 600.

        Returns:
            str: Generated text content from the completion model.

        Raises:
            Exception: Propagates unhandled completion errors.
        """
        response = await self.client.chat.completions.create(
            model=self.small_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        if getattr(response, "usage", None) is not None:
            logger.info(f"[TOKENS] purpose={purpose.value} model={self.small_model} prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens}")
        else:
            logger.warning(f"[TOKENS] purpose={purpose.value} Usage statistics were missing or None.")

        return response.choices[0].message.content
    