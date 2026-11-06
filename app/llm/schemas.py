from typing import Literal

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1000, gt=0)


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: dict[str, int] | None = None