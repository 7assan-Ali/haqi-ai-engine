from app.core.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.mock_provider import MockLLMProvider
from app.llm.openai_provider import OpenAIProvider


def get_llm_provider() -> BaseLLMProvider:

    provider_name = settings.LLM_PROVIDER.lower()

    if provider_name == "mock":
        return MockLLMProvider()

    if provider_name == "openai":
        return OpenAIProvider()

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}"
    )