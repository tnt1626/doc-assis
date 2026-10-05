from enum import Enum
from app.services.agent.tools import TOOLS
from app.services.llm.openai_compat import OpenAICompatClient
from app.services.llm.clients import groq_client, ollama_openai_client
from app.config import GROQ_MODEL, GROQ_SMALL_MODEL, OLLAMA_MODEL, LLM_PROVIDER


class Provider(str, Enum):
    """Supported LLM provider types."""
    OLLAMA  = "ollama"
    GROQ    = "groq"

def get_llm(provider: Provider):
    """Instantiate and return an LLMClient based on the specified provider."""
    if provider == "ollama":
        return OpenAICompatClient(
            client=ollama_openai_client,
            model=OLLAMA_MODEL,
            small_model=OLLAMA_MODEL,
            tools=TOOLS,
            provider_options={
                "stream_options": {"include_usage": True}
            },
        )
    elif provider == "groq":
        return OpenAICompatClient(
            client=groq_client,
            model=GROQ_MODEL,
            small_model=GROQ_SMALL_MODEL,
            tools=TOOLS
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


llm = get_llm(Provider(LLM_PROVIDER))


async def main():
    from app.services.llm.llm_types import LLMPurpose
    async for ev in llm.stream(
        [{"role": "user", "content": "hello"}],
        LLMPurpose.AGENT
    ):
        print(ev)

    text = await llm.completion(
        [{"role": "user", "content": "hello"}],
        LLMPurpose.AGENT
    )
    print(text)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())