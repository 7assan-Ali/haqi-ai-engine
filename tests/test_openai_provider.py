
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings
from app.llm.openai_provider import OpenAIProvider
from app.llm.schemas import LLMMessage, LLMRequest


@pytest.mark.asyncio
async def test_openai_generate():

    settings.LLM_API_KEY = "test-api-key"
    settings.LLM_MODEL = "test-model"

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Test answer from OpenAI"
                )
            )
        ],
        model="test-model",
        usage=SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        ),
    )

    with patch(
        "app.llm.openai_provider.AsyncOpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create = AsyncMock(
            return_value=fake_response
        )

        provider = OpenAIProvider()

        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="What is Haqi?"
                )
            ]
        )

        response = await provider.generate(request)

        assert response.content == "Test answer from OpenAI"
        assert response.provider == "openai"
        assert response.model == "test-model"
        assert response.usage["total_tokens"] == 15



def test_openai_provider_requires_api_key(monkeypatch):

    monkeypatch.setattr(settings, "LLM_API_KEY", "")

    with pytest.raises(
        ValueError,
        match="LLM_API_KEY is missing"
    ):
        OpenAIProvider()
