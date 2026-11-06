
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.llm.factory import get_llm_provider
from app.llm.mock_provider import MockLLMProvider


def test_factory_returns_mock_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    provider = get_llm_provider()

    assert isinstance(provider, MockLLMProvider)


def test_factory_returns_openai_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")

    with patch("app.llm.factory.OpenAIProvider") as mock_provider:
        get_llm_provider()

        mock_provider.assert_called_once()


def test_factory_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "unsupported")

    with pytest.raises(
        ValueError,
        match="Unsupported LLM provider"
    ):
        get_llm_provider()
