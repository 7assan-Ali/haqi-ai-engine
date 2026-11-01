from app.llm.base import BaseLLMProvider
from app.llm.schemas import LLMRequest, LLMResponse


class MockLLMProvider(BaseLLMProvider):

    async def generate(self, request: LLMRequest) -> LLMResponse:
        last_message = request.messages[-1].content

        return LLMResponse(
            content=f"Mock response from Haqi: {last_message}",
            provider="mock",
            model="mock-model",
            usage=None
        )