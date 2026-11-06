from openai import AsyncOpenAI

from app.core.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.schemas import LLMRequest, LLMResponse


class OpenAIProvider(BaseLLMProvider):

    def __init__(self):
        if not settings.LLM_API_KEY:
            raise ValueError("LLM_API_KEY is missing from environment settings.")

        client_config = {
            "api_key": settings.LLM_API_KEY,
        }

        if settings.LLM_BASE_URL:
            client_config["base_url"] = settings.LLM_BASE_URL

        self.client = AsyncOpenAI(**client_config)

    async def generate(self, request: LLMRequest) -> LLMResponse:

        response = await self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                message.model_dump()
                for message in request.messages
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            provider="openai",
            model=response.model,
            usage=(
                response.usage.model_dump()
                if response.usage
                else None
            ),
        )